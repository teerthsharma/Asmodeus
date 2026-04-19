import logging
from typing import Iterable, List, Optional, Set

from .task import Task
from .skills import SkillLike, coerce_skill_names

logger = logging.getLogger(__name__)

class WorkerAgent:
    """
    Specialized agents tasked with resolving subtasks assigned by the TaskManager.
    Includes built-in transient error recovery and retry logic.
    """
    def __init__(
        self,
        name: str,
        max_retries: int = 3,
        skills: Optional[Iterable[SkillLike]] = None,
    ):
        self.name = name
        self.max_retries = max_retries
        self.skills: Set[str] = coerce_skill_names(skills)
        self.subagents: List["WorkerAgent"] = []

    def register_subagent(self, subagent: "WorkerAgent") -> None:
        """Attaches a subagent under this worker for specialized execution."""

        self.subagents.append(subagent)

    def spawn_subagent(
        self,
        name: str,
        skills: Optional[Iterable[SkillLike]] = None,
        max_retries: Optional[int] = None,
    ) -> "WorkerAgent":
        """Convenience constructor to create and register a specialized subagent."""

        subagent = WorkerAgent(
            name=name,
            max_retries=self.max_retries if max_retries is None else max_retries,
            skills=skills,
        )
        self.register_subagent(subagent)
        return subagent

    def get_all_agents(self) -> List["WorkerAgent"]:
        """Returns this worker and all descendant subagents as a flat list."""

        agents: List[WorkerAgent] = [self]
        for subagent in self.subagents:
            agents.extend(subagent.get_all_agents())
        return agents

    def skill_match_score(self, task: Task) -> int:
        """Returns the number of required task skills this worker can satisfy."""

        if not task.requires_skill_match():
            return 0
        return len(task.required_skills.intersection(self.skills))

    def can_handle_task(self, task: Task) -> bool:
        """True when this worker can satisfy all required skills for the task."""

        return task.matches_agent_skills(self.skills)

    def execute_task(self, task: Task) -> bool:
        """
        Executes a given task with automated retries.
        """
        retries = 0
        
        while retries <= self.max_retries:
            try:
                result = task.execute()
                task.verify(result)
                logger.info(f"Worker '{self.name}' successfully executed task: {result}")
                return True
            except Exception as e:
                task.log_failure(e)
                retries += 1
                if retries <= self.max_retries:
                    logger.warning(f"Worker '{self.name}' encountered an error on task '{task.name}'. Retrying ({retries}/{self.max_retries}): {e}")
                else:
                    logger.error(f"Worker '{self.name}' exhausted all retries for task '{task.name}'. Final error: {e}")
        
        return False
