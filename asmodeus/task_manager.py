import time
from typing import List, Dict, Optional
from .scout import ScoutAgent
from .worker import WorkerAgent
from .task import Task
from .telemetry import TelemetryCollector, FallbackEvent

class TaskManager:
    """
    Responsible for orchestrating the swarm: assigning tasks discovered 
    by scouts to the allocated worker agents and logging fallback telemetry.
    """
    def __init__(self, telemetry_collector: Optional[TelemetryCollector] = None):
        self.workers: List[WorkerAgent] = []
        self.telemetry = telemetry_collector or TelemetryCollector()

    def register_worker(self, worker: WorkerAgent) -> None:
        """
        Adds a worker to the available pool.
        """
        self.workers.append(worker)

    def create_subagent(
        self,
        parent_worker_name: str,
        subagent_name: str,
        skills: List[str] = None,
        max_retries: int = None,
    ) -> WorkerAgent:
        """Creates a subagent under a named parent worker."""

        for worker in self.workers:
            if worker.name == parent_worker_name:
                return worker.spawn_subagent(
                    name=subagent_name,
                    skills=skills,
                    max_retries=max_retries,
                )

        raise ValueError(f"Parent worker '{parent_worker_name}' is not registered.")

    def get_available_workers(self) -> List[WorkerAgent]:
        """Returns all routable workers, including registered subagents."""

        expanded_workers: List[WorkerAgent] = []
        for worker in self.workers:
            expanded_workers.extend(worker.get_all_agents())
        return expanded_workers

    def assign_tasks_from_scout(self, scout: ScoutAgent) -> Dict[Task, List[WorkerAgent]]:
        """
        Orchestrates task assignment from a scout's discovery to the workers.
        """
        tasks = scout.discover_tasks()
        assignments = {}
        available_workers = self.get_available_workers()
        
        for task in tasks:
            allocated_agents = scout.route_task_to_agents(task, available_workers)
            assignments[task] = allocated_agents
            
        return assignments

    def execute_all(self, assignments: Dict[Task, List[WorkerAgent]]) -> Dict[str, bool]:
        """
        Runs all assigned tasks with their respective workers, capturing fallback telemetry on failures/reroutes.
        """
        results = {}
        for task, assigned_workers in assignments.items():
            if not assigned_workers:
                self.telemetry.record_fallback(FallbackEvent(
                    task_name=task.name,
                    trigger_reason="No assigned workers available",
                    failed_specialist="None",
                    chosen_backup=None,
                    retry_count=0,
                ))
                results[task.name] = False
                continue

            success = False
            prev_worker = None

            for idx, worker in enumerate(assigned_workers):
                if idx > 0 and prev_worker is not None:
                    # Rerouted to a secondary/backup worker
                    start_time = time.perf_counter()
                    # Log fallback transition
                    latency_ms = (time.perf_counter() - start_time) * 1000.0
                    self.telemetry.record_fallback(FallbackEvent(
                        task_name=task.name,
                        trigger_reason=f"Primary worker {prev_worker.name} failed task",
                        failed_specialist=prev_worker.name,
                        chosen_backup=worker.name,
                        retry_count=task.failures,
                        reroute_latency_ms=latency_ms,
                    ))

                if worker.execute_task(task):
                    success = True
                    break
                else:
                    # Execution failed on this worker
                    self.telemetry.record_fallback(FallbackEvent(
                        task_name=task.name,
                        trigger_reason=f"Worker {worker.name} exhausted retries",
                        failed_specialist=worker.name,
                        chosen_backup=assigned_workers[idx + 1].name if idx + 1 < len(assigned_workers) else None,
                        retry_count=task.failures,
                    ))
                    prev_worker = worker

            results[task.name] = success
            
        return results
