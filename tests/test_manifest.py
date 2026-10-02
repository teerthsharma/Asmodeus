import unittest
import tempfile
from pathlib import Path

from asmodeus.manifest import ArtifactManifest
from asmodeus.registry import Registry, ModelStatus
from asmodeus.downloader import SLMDownloader, VerificationError


class TestArtifactManifestAndAdmission(unittest.TestCase):
    def test_manifest_validation_valid(self):
        manifest = ArtifactManifest(
            source="huggingface/Qwen/Qwen2.5-1.5B-Instruct",
            revision_pin="v1.0.0",
            checksum="abc123hash",
            license="apache-2.0",
            compatibility={"python": ">=3.10", "cuda": "12.1"},
        )
        self.assertEqual(manifest.source, "huggingface/Qwen/Qwen2.5-1.5B-Instruct")
        self.assertEqual(manifest.revision_pin, "v1.0.0")

    def test_manifest_validation_missing_field(self):
        with self.assertRaises(ValueError):
            ArtifactManifest(
                source="",
                revision_pin="v1.0.0",
                checksum="abc123hash",
                license="apache-2.0",
            )

    def test_manifest_from_dict(self):
        data = {
            "source": "models/qwen",
            "revision_pin": "main",
            "checksum": "12345",
            "license": "mit",
            "compatibility": {},
        }
        manifest = ArtifactManifest.from_dict(data)
        self.assertEqual(manifest.checksum, "12345")

    def test_downloader_with_manifest_verification(self):
        registry = Registry()
        registry.register_model("qwen-nano", 1_000_000, ["coding"])

        with tempfile.TemporaryDirectory() as tmp_dir:
            downloader = SLMDownloader(registry, cache_dir=tmp_dir)

            valid_manifest = ArtifactManifest(
                source="test_source",
                revision_pin="v1",
                checksum="mock_hash",
                license="mit",
            )

            # File hash for b"mock_model_data_stream"
            import hashlib
            expected_hash = hashlib.sha256(b"mock_model_data_stream").hexdigest()

            downloader.add_trusted_source(
                "qwen-nano",
                expected_hash=expected_hash,
                provenance="trusted_repo",
                manifest=valid_manifest,
            )

            success = downloader.download("qwen-nano", "http://valid.url/model.bin")
            self.assertTrue(success)

            model = registry.get_model("qwen-nano")
            self.assertIsNotNone(model)
            self.assertEqual(model.status, ModelStatus.READY)
            self.assertIsNotNone(model.manifest)
            self.assertEqual(model.manifest.license, "mit")

    def test_downloader_unregistered_model(self):
        registry = Registry()
        with tempfile.TemporaryDirectory() as tmp_dir:
            downloader = SLMDownloader(registry, cache_dir=tmp_dir)
            self.assertFalse(downloader.download("nonexistent-model", "http://url/m.bin"))

    def test_downloader_checksum_mismatch(self):
        registry = Registry()
        registry.register_model("qwen-nano", 1_000_000, ["coding"])
        with tempfile.TemporaryDirectory() as tmp_dir:
            downloader = SLMDownloader(registry, cache_dir=tmp_dir)
            downloader.add_trusted_source(
                "qwen-nano",
                expected_hash="wrong_hash_123",
                provenance="trusted_repo",
            )
            success = downloader.download("qwen-nano", "http://valid.url/model.bin")
            self.assertFalse(success)
            model = registry.get_model("qwen-nano")
            self.assertEqual(model.status, ModelStatus.FAILED)

    def test_downloader_untrusted_provenance(self):
        registry = Registry()
        registry.register_model("qwen-nano", 1_000_000, ["coding"])
        import hashlib
        expected_hash = hashlib.sha256(b"mock_model_data_stream").hexdigest()

        with tempfile.TemporaryDirectory() as tmp_dir:
            downloader = SLMDownloader(registry, cache_dir=tmp_dir)
            downloader.add_trusted_source(
                "qwen-nano",
                expected_hash=expected_hash,
                provenance="untrusted_source",
            )
            success = downloader.download("qwen-nano", "http://valid.url/model.bin")
            self.assertFalse(success)
            model = registry.get_model("qwen-nano")
            self.assertEqual(model.status, ModelStatus.FAILED)

    def test_downloader_fallback_success(self):
        registry = Registry()
        registry.register_model("qwen-nano", 1_000_000, ["coding"])

        with tempfile.TemporaryDirectory() as tmp_dir:
            downloader = SLMDownloader(registry, cache_dir=tmp_dir)
            # Pre-create backup file
            backup_file = Path(tmp_dir) / "qwen-nano.bin.backup"
            backup_file.write_bytes(b"backup_data_stream")

            import hashlib
            expected_hash = hashlib.sha256(b"backup_data_stream").hexdigest()
            downloader.add_trusted_source(
                "qwen-nano",
                expected_hash=expected_hash,
                provenance="trusted_backup",
            )

            # URL containing 'invalid' triggers primary fetch failure and degraded fallback
            success = downloader.download("qwen-nano", "http://invalid.url/model.bin")
            self.assertTrue(success)
            self.assertEqual(registry.get_model("qwen-nano").status, ModelStatus.READY)


if __name__ == "__main__":
    unittest.main()
