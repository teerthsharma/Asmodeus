from typing import List
from .task import Task
from .worker import WorkerAgent

class ScoutAgent:
    """
    Responsible for analyzing the environment, discovering objectives, 
    and generating tasks for WorkerAgents.
    """
    def __init__(self, name: str):
        self.name = name

    def discover_tasks(self) -> List[Task]:
        """
        Analyze environment and generate a list of subtasks.
        """
        return [
            Task(name="Initial System Recon"),
            Task(name="Data Gathering", should_fail_times=2),
            Task(name="Final Topology Report")
        ]

    def route_task_to_agents(self, task: Task, available_workers: List[WorkerAgent]) -> List[WorkerAgent]:
        """
        Determines the optimal workers for a specific task based on capabilities.
        """
        if not available_workers:
            return []

        if not task.requires_skill_match():
            return [available_workers[0]]

        candidates = [worker for worker in available_workers if worker.can_handle_task(task)]
        if not candidates:
            return []

        return sorted(candidates, key=lambda worker: worker.skill_match_score(task), reverse=True)
