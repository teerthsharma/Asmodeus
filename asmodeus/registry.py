from typing import Dict, List, Set, Optional
from dataclasses import dataclass, field
from enum import Enum

class ModelStatus(Enum):
    PENDING = "pending"
    DOWNLOADING = "downloading"
    VERIFIED = "verified"
    READY = "ready"
    FAILED = "failed"
    UNAVAILABLE = "unavailable"

@dataclass
class MicroMaster:
    """
    Represents a specialist micro-master model in the swarm.
    Tier A (Nano): ~1M parameters
    Tier B (Micro): ~10M parameters
    """
    model_id: str
    parameter_count: int  # e.g., 1_000_000 or 10_000_000
    capability_tags: Set[str] = field(default_factory=set)
    status: ModelStatus = ModelStatus.PENDING
    reliability: float = 1.0  # Historical reliability [0.0, 1.0]
    cost_per_call: float = 1.0 # Cost metric

class Registry:
    """
    Registry to manage heterogeneous 1M/10M micro-masters.
    Maintains the static topology budget and tracks M_ready(t).
    """
    def __init__(self):
        self.models: Dict[str, MicroMaster] = {}

    def register_model(self, model_id: str, parameter_count: int, tags: List[str]) -> None:
        """Registers a new model into the static topology."""
        if model_id not in self.models:
            self.models[model_id] = MicroMaster(
                model_id=model_id,
                parameter_count=parameter_count,
                capability_tags=set(tags)
            )

    def update_status(self, model_id: str, status: ModelStatus) -> None:
        """Updates the status of a registered model."""
        if model_id in self.models:
            self.models[model_id].status = status

    def get_ready_models(self) -> List[MicroMaster]:
        """
        Returns the admission set M_ready(t).
        M_ready(t) = {m in M_internet | downloaded(m,t) and verified(m) and compatible(m)}
        """
        return [m for m in self.models.values() if m.status == ModelStatus.READY]

    def get_models_by_capability(self, capability: str) -> List[MicroMaster]:
        """Filters ready models by a specific capability tag."""
        return [m for m in self.get_ready_models() if capability in m.capability_tags]

    def get_model(self, model_id: str) -> Optional[MicroMaster]:
        """Retrieves a model by its ID."""
        return self.models.get(model_id)

    def total_ready_parameters(self) -> int:
        """
        Computes the active parameter budget.
        P_total = sum_{i=1..N} P_i for ready models.
        """
        return sum(m.parameter_count for m in self.get_ready_models())
