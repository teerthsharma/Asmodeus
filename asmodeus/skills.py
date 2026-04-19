from dataclasses import dataclass
from typing import Iterable, Optional, Set, Union


@dataclass(frozen=True)
class Skill:
    """Represents a named capability that can be attached to workers and tasks."""

    name: str


def normalize_skill_name(name: str) -> str:
    """Creates a canonical representation so routing checks are consistent."""

    return name.strip().lower().replace(" ", "_")


SkillLike = Union[str, Skill]


def coerce_skill_names(skills: Optional[Iterable[SkillLike]]) -> Set[str]:
    """Normalizes mixed skill inputs (Skill objects or strings) into a set."""

    if not skills:
        return set()

    normalized: Set[str] = set()
    for skill in skills:
        raw_name = skill.name if isinstance(skill, Skill) else str(skill)
        canonical = normalize_skill_name(raw_name)
        if canonical:
            normalized.add(canonical)

    return normalized
