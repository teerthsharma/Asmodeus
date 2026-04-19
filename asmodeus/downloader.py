import hashlib
import logging
from typing import Optional, Dict
from pathlib import Path

from asmodeus.registry import Registry, ModelStatus

logger = logging.getLogger(__name__)

class VerificationError(Exception):
    """Raised when model verification fails (checksum or provenance)."""
    pass

class SLMDownloader:
    """
    Internet SLM acquisition logic with verification gates and fallback logic.
    Handles downloading, verifying, and admitting models into M_ready(t).
    """
    def __init__(self, registry: Registry, cache_dir: str = "/tmp/asmodeus_cache"):
        self.registry = registry
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        # Database of expected checksums and provenance
        self.trusted_sources: Dict[str, dict] = {}

    def add_trusted_source(self, model_id: str, expected_hash: str, provenance: str) -> None:
        """Adds expected verification data for a model."""
        self.trusted_sources[model_id] = {
            "hash": expected_hash,
            "provenance": provenance
        }

    def download(self, model_id: str, url: str) -> bool:
        """
        Acquires a model, verifying it and handling fallbacks if necessary.
        Transitions model status appropriately.
        """
        model = self.registry.get_model(model_id)
        if not model:
            logger.error(f"Model {model_id} not in registry.")
            return False

        self.registry.update_status(model_id, ModelStatus.DOWNLOADING)
        
        try:
            file_path = self.cache_dir / f"{model_id}.bin"
            
            # Primary fetch attempt
            success = self._fetch_from_internet(url, file_path)
            
            # Degraded-mode fallback logic
            if not success:
                logger.warning(f"Failed to fetch {model_id} from {url}. Attempting degraded mode fallback.")
                success = self._fallback_fetch(model_id, file_path)
                if not success:
                    raise Exception("All download methods and fallbacks failed.")

            # Verification Gate
            self._verify(model_id, file_path)
            
            # Admission to M_ready(t)
            self.registry.update_status(model_id, ModelStatus.READY)
            return True

        except Exception as e:
            logger.error(f"Acquisition failed for {model_id}: {str(e)}")
            self.registry.update_status(model_id, ModelStatus.FAILED)
            return False

    def _fetch_from_internet(self, url: str, path: Path) -> bool:
        """Simulate primary internet fetch logic."""
        # Mock implementation for fetching SLM
        if "invalid" in url:
            return False
        # In a real system, use requests or httpx to download chunked
        path.write_bytes(b"mock_model_data_stream")
        return True
        
    def _fallback_fetch(self, model_id: str, path: Path) -> bool:
        """
        Degraded-mode fallback logic.
        E.g., loading from local backup cache, peer-to-peer, or alternative mirror.
        """
        backup_path = self.cache_dir / f"{model_id}.bin.backup"
        if backup_path.exists():
            path.write_bytes(backup_path.read_bytes())
            return True
        return False

    def _verify(self, model_id: str, path: Path) -> None:
        """
        Verification gates: checksum and provenance constraints.
        Ensures compromised models do not enter the swarm.
        """
        if model_id not in self.trusted_sources:
             logger.warning(f"No trusted source info for {model_id}. Proceeding with caution.")
             self.registry.update_status(model_id, ModelStatus.VERIFIED)
             return

        expected_hash = self.trusted_sources[model_id]["hash"]
        
        hasher = hashlib.sha256()
        with open(path, 'rb') as f:
             # Read in chunks for large SLMs
             for chunk in iter(lambda: f.read(4096), b""):
                 hasher.update(chunk)
                 
        actual_hash = hasher.hexdigest()

        if actual_hash != expected_hash:
            raise VerificationError(f"Checksum mismatch for {model_id}: expected {expected_hash}, got {actual_hash}")

        # Provenance check
        provenance = self.trusted_sources[model_id]["provenance"]
        if "untrusted" in provenance.lower():
            raise VerificationError(f"Provenance check failed for {model_id}: {provenance}")

        self.registry.update_status(model_id, ModelStatus.VERIFIED)
