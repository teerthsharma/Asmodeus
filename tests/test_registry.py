import unittest
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from asmodeus.registry import Registry, MicroMaster, ModelStatus


class TestRegistry(unittest.TestCase):
    def setUp(self):
        self.registry = Registry()

    def test_register_and_get_model(self):
        self.registry.register_model("model-a", 1_000_000, ["math", "logic"])
        model = self.registry.get_model("model-a")

        self.assertIsNotNone(model)
        self.assertEqual(model.model_id, "model-a")
        self.assertEqual(model.parameter_count, 1_000_000)
        self.assertIn("math", model.capability_tags)
        self.assertEqual(model.status, ModelStatus.PENDING)

    def test_duplicate_registration_ignored(self):
        self.registry.register_model("model-a", 1_000_000, ["math"])
        self.registry.register_model("model-a", 2_000_000, ["code"])
        model = self.registry.get_model("model-a")

        self.assertEqual(model.parameter_count, 1_000_000)

    def test_update_status_and_ready_models(self):
        self.registry.register_model("model-a", 1_000_000, ["math"])
        self.registry.register_model("model-b", 10_000_000, ["code"])

        self.assertEqual(len(self.registry.get_ready_models()), 0)

        self.registry.update_status("model-a", ModelStatus.READY)
        ready = self.registry.get_ready_models()
        self.assertEqual(len(ready), 1)
        self.assertEqual(ready[0].model_id, "model-a")

    def test_get_models_by_capability(self):
        self.registry.register_model("model-a", 1_000_000, ["math", "logic"])
        self.registry.register_model("model-b", 10_000_000, ["code", "logic"])

        self.registry.update_status("model-a", ModelStatus.READY)
        self.registry.update_status("model-b", ModelStatus.READY)

        logic_models = self.registry.get_models_by_capability("logic")
        code_models = self.registry.get_models_by_capability("code")

        self.assertEqual(len(logic_models), 2)
        self.assertEqual(len(code_models), 1)

    def test_total_ready_parameters(self):
        self.registry.register_model("model-a", 1_000_000, ["math"])
        self.registry.register_model("model-b", 10_000_000, ["code"])

        self.assertEqual(self.registry.total_ready_parameters(), 0)

        self.registry.update_status("model-a", ModelStatus.READY)
        self.assertEqual(self.registry.total_ready_parameters(), 1_000_000)

        self.registry.update_status("model-b", ModelStatus.READY)
        self.assertEqual(self.registry.total_ready_parameters(), 11_000_000)


if __name__ == "__main__":
    unittest.main()
