from dataclasses import dataclass, field
from typing import Dict, Any, Optional

REQUIRED_FIELDS = ("source", "revision_pin", "checksum", "license", "compatibility")

@dataclass
class ArtifactManifest:
    """
    Schema for specialist model artifact manifests.
    Ensures source, revision pinning, integrity checksum, license compliance,
    and runtime compatibility before model admission into M_ready(t).
    """
    source: str
    revision_pin: str
    checksum: str
    license: str
    compatibility: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        self.validate()

    def validate(self) -> None:
        """
        Validates all required manifest fields.
        Raises ValueError if any field is missing, empty, or invalid.
        """
        for field_name in REQUIRED_FIELDS:
            value = getattr(self, field_name, None)
            if value is None or (isinstance(value, str) and not value.strip()):
                raise ValueError(f"Manifest missing or empty required field: '{field_name}'")

        if not isinstance(self.compatibility, dict):
            raise ValueError("Manifest 'compatibility' field must be a dictionary.")

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ArtifactManifest":
        """
        Constructs an ArtifactManifest from a dictionary.
        Raises ValueError if any required field is absent.
        """
        if not isinstance(data, dict):
            raise ValueError("Manifest data must be a dictionary.")

        missing = [f for f in REQUIRED_FIELDS if f not in data]
        if missing:
            raise ValueError(f"Manifest missing required fields: {', '.join(missing)}")

        return cls(
            source=data["source"],
            revision_pin=data["revision_pin"],
            checksum=data["checksum"],
            license=data["license"],
            compatibility=data.get("compatibility", {}),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source": self.source,
            "revision_pin": self.revision_pin,
            "checksum": self.checksum,
            "license": self.license,
            "compatibility": self.compatibility,
        }
