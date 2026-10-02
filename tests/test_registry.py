import unittest

from asmodeus.manifest import ArtifactManifest
from asmodeus.registry import Registry, ModelStatus


class TestRegistry(unittest.TestCase):
    def setUp(self):
        self.registry = Registry()

    def test_register_and_get_model(self):
        self.registry.register_model("model-1", 1_000_000, ["coding", "python"])
        model = self.registry.get_model("model-1")
        self.assertIsNotNone(model)
        self.assertEqual(model.parameter_count, 1_000_000)
        self.assertIn("coding", model.capability_tags)
        self.assertEqual(model.status, ModelStatus.PENDING)

    def test_get_ready_models_and_total_parameters(self):
        self.registry.register_model("m1", 1_000_000, ["coding"])
        self.registry.register_model("m2", 10_000_000, ["math"])
        self.registry.register_model("m3", 1_000_000, ["logic"])

        self.registry.update_status("m1", ModelStatus.READY)
        self.registry.update_status("m2", ModelStatus.READY)
        self.registry.update_status("m3", ModelStatus.FAILED)

        ready = self.registry.get_ready_models()
        self.assertEqual(len(ready), 2)
        self.assertEqual(self.registry.total_ready_parameters(), 11_000_000)

    def test_get_models_by_capability(self):
        self.registry.register_model("m1", 1_000_000, ["coding", "python"])
        self.registry.register_model("m2", 10_000_000, ["coding", "c++"])
        self.registry.update_status("m1", ModelStatus.READY)
        self.registry.update_status("m2", ModelStatus.READY)

        python_models = self.registry.get_models_by_capability("python")
        self.assertEqual(len(python_models), 1)
        self.assertEqual(python_models[0].model_id, "m1")

    def test_attach_manifest_dict_and_object(self):
        self.registry.register_model("m1", 1_000_000, ["coding"])
        manifest_dict = {
            "source": "repo/m1",
            "revision_pin": "v1.0",
            "checksum": "abc",
            "license": "mit",
            "compatibility": {},
        }
        self.registry.attach_manifest("m1", manifest_dict)
        model = self.registry.get_model("m1")
        self.assertIsNotNone(model.manifest)
        self.assertEqual(model.manifest.source, "repo/m1")

    def test_get_ready_models_filters_out_invalid_manifest(self):
        manifest = ArtifactManifest(
            source="repo/m1",
            revision_pin="v1.0",
            checksum="abc",
            license="mit",
        )
        self.registry.register_model("m1", 1_000_000, ["coding"], manifest=manifest)
        self.registry.update_status("m1", ModelStatus.READY)

        # Mutate manifest required field to make it invalid
        self.registry.get_model("m1").manifest.source = ""
        ready = self.registry.get_ready_models()
        self.assertEqual(len(ready), 0)


if __name__ == "__main__":
    unittest.main()
