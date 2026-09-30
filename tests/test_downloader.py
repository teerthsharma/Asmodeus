import hashlib
import tempfile
import unittest
import sys
import os
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from asmodeus.downloader import SLMDownloader, VerificationError
from asmodeus.registry import Registry, ModelStatus


class TestSLMDownloader(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.registry = Registry()
        self.downloader = SLMDownloader(registry=self.registry, cache_dir=self.temp_dir.name)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_download_unregistered_model(self):
        success = self.downloader.download("nonexistent-model", "http://example.com/model.bin")
        self.assertFalse(success)

    def test_successful_download_without_verification_info(self):
        self.registry.register_model("model-a", 1_000_000, ["math"])
        success = self.downloader.download("model-a", "http://example.com/model.bin")

        self.assertTrue(success)
        model = self.registry.get_model("model-a")
        self.assertEqual(model.status, ModelStatus.READY)

    def test_download_with_valid_verification_and_provenance(self):
        self.registry.register_model("model-verified", 1_000_000, ["math"])

        # Expected hash for "mock_model_data_stream"
        expected_data = b"mock_model_data_stream"
        expected_hash = hashlib.sha256(expected_data).hexdigest()

        self.downloader.add_trusted_source(
            model_id="model-verified",
            expected_hash=expected_hash,
            provenance="trusted_source_org"
        )

        success = self.downloader.download("model-verified", "http://example.com/valid.bin")
        self.assertTrue(success)
        model = self.registry.get_model("model-verified")
        self.assertEqual(model.status, ModelStatus.READY)

    def test_download_fails_checksum_mismatch(self):
        self.registry.register_model("model-bad-hash", 1_000_000, ["math"])

        self.downloader.add_trusted_source(
            model_id="model-bad-hash",
            expected_hash="invalid_hash_12345",
            provenance="trusted_source_org"
        )

        success = self.downloader.download("model-bad-hash", "http://example.com/valid.bin")
        self.assertFalse(success)
        model = self.registry.get_model("model-bad-hash")
        self.assertEqual(model.status, ModelStatus.FAILED)

    def test_download_fails_untrusted_provenance(self):
        self.registry.register_model("model-untrusted", 1_000_000, ["math"])

        expected_data = b"mock_model_data_stream"
        expected_hash = hashlib.sha256(expected_data).hexdigest()

        self.downloader.add_trusted_source(
            model_id="model-untrusted",
            expected_hash=expected_hash,
            provenance="untrusted_vendor"
        )

        success = self.downloader.download("model-untrusted", "http://example.com/valid.bin")
        self.assertFalse(success)
        model = self.registry.get_model("model-untrusted")
        self.assertEqual(model.status, ModelStatus.FAILED)

    def test_degraded_mode_fallback_recovery(self):
        self.registry.register_model("model-fallback", 1_000_000, ["math"])

        # Create backup cache file
        cache_path = Path(self.temp_dir.name)
        backup_file = cache_path / "model-fallback.bin.backup"
        backup_file.write_bytes(b"mock_backup_data")

        # Force internet download failure via invalid URL
        success = self.downloader.download("model-fallback", "http://invalid-url.com")
        self.assertTrue(success)
        model = self.registry.get_model("model-fallback")
        self.assertEqual(model.status, ModelStatus.READY)


if __name__ == "__main__":
    unittest.main()
