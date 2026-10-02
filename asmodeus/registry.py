from typing import Dict, List, Set, Optional, Union
from dataclasses import dataclass, field
from enum import Enum

from asmodeus.manifest import ArtifactManifest

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
    manifest: Optional[ArtifactManifest] = None

class Registry:
    """
    Registry to manage heterogeneous 1M/10M micro-masters.
    Maintains the static topology budget and tracks M_ready(t).
    """
    def __init__(self):
        self.models: Dict[str, MicroMaster] = {}

    def register_model(
        self,
        model_id: str,
        parameter_count: int,
        tags: List[str],
        manifest: Optional[Union[ArtifactManifest, Dict]] = None,
    ) -> None:
        """Registers a new model into the static topology."""
        validated_manifest = None
        if manifest is not None:
            if isinstance(manifest, ArtifactManifest):
                validated_manifest = manifest
            elif isinstance(manifest, dict):
                validated_manifest = ArtifactManifest.from_dict(manifest)

        if model_id not in self.models:
            self.models[model_id] = MicroMaster(
                model_id=model_id,
                parameter_count=parameter_count,
                capability_tags=set(tags),
                manifest=validated_manifest,
            )

    def attach_manifest(self, model_id: str, manifest: Union[ArtifactManifest, Dict]) -> None:
        """Attaches and validates a manifest for a registered model."""
        if model_id in self.models:
            if isinstance(manifest, ArtifactManifest):
                manifest.validate()
                self.models[model_id].manifest = manifest
            elif isinstance(manifest, dict):
                self.models[model_id].manifest = ArtifactManifest.from_dict(manifest)

    def update_status(self, model_id: str, status: ModelStatus) -> None:
        """Updates the status of a registered model."""
        if model_id in self.models:
            self.models[model_id].status = status

    def get_ready_models(self) -> List[MicroMaster]:
        """
        Returns the admission set M_ready(t).
        M_ready(t) = {m in M_internet | downloaded(m,t) and verified(m) and compatible(m)}
        Enforces that status is ModelStatus.READY.
        """
        ready_models = []
        for m in self.models.values():
            if m.status == ModelStatus.READY:
                # If a manifest exists, validate it
                if m.manifest is not None:
                    try:
                        m.manifest.validate()
                    except ValueError:
                        continue
                ready_models.append(m)
        return ready_models

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
