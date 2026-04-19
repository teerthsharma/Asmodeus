from typing import List, Dict
from .scout import ScoutAgent
from .worker import WorkerAgent
from .task import Task

class TaskManager:
    """
    Responsible for orchestrating the swarm: assigning tasks discovered 
    by scouts to the allocated worker agents.
    """
    def __init__(self):
        self.workers: List[WorkerAgent] = []

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
        Runs all assigned tasks with their respective workers.
        """
        results = {}
        for task, assigned_workers in assignments.items():
            if not assigned_workers:
                results[task.name] = False
                continue

            success = False
            for worker in assigned_workers:
                if worker.execute_task(task):
                    success = True
                    break

            results[task.name] = success
            
        return results
