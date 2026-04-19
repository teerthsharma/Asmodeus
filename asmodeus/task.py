from typing import Any, Iterable, Optional

from .skills import SkillLike, coerce_skill_names

class Task:
    """
    Represents a specific objective or task to be executed by a WorkerAgent.
    """
    def __init__(
        self,
        name: str,
        should_fail_times: int = 0,
        required_skills: Optional[Iterable[SkillLike]] = None,
    ):
        self.name = name
        self.should_fail_times = should_fail_times
        self.required_skills = coerce_skill_names(required_skills)
        self.failures = 0
        self.executed = False

    def requires_skill_match(self) -> bool:
        """Indicates whether routing must satisfy explicit skill requirements."""

        return bool(self.required_skills)

    def matches_agent_skills(self, agent_skills: Iterable[str]) -> bool:
        """Checks if the provided skill set can satisfy this task."""

        if not self.required_skills:
            return True
        normalized_agent_skills = coerce_skill_names(agent_skills)
        return self.required_skills.issubset(normalized_agent_skills)

    def execute(self) -> str:
        """
        Executes the task. Simulates failures if configured.
        
        Raises:
            Exception: If the task simulates a transient failure.
            
        Returns:
            str: A completion message on success.
        """
        if self.failures < self.should_fail_times:
            self.failures += 1
            raise Exception(f"Simulated transient failure on task '{self.name}' (Failure #{self.failures})")
        
        self.executed = True
        return f"Task '{self.name}' executed successfully"

    def verify(self, result: Any) -> bool:
        """
        Verifies the output of the task.
        """
        return self.executed

    def log_failure(self, error: Exception) -> None:
        """
        Logs a task failure.
        """
        pass # In a real system, this would log to a database or file
