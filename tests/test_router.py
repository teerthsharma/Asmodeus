import unittest
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from asmodeus.registry import Registry, ModelStatus
from asmodeus.router import SparseTopKRouter


class TestSparseTopKRouter(unittest.TestCase):
    def setUp(self):
        self.registry = Registry()

        # Register several models with different tags, parameter counts, cost and reliability
        self.registry.register_model("model-math", 1_000_000, ["Math", "Calculus"])
        self.registry.register_model("model-code", 2_000_000, ["Code", "Python"])
        self.registry.register_model("model-general", 5_000_000, ["Math", "Code", "General"])
        self.registry.register_model("model-pending", 1_000_000, ["Math"])

        # Mark ready models
        self.registry.update_status("model-math", ModelStatus.READY)
        self.registry.update_status("model-code", ModelStatus.READY)
        self.registry.update_status("model-general", ModelStatus.READY)
        # model-pending stays PENDING

        self.router = SparseTopKRouter(registry=self.registry, k=2)

    def test_routing_filters_unready_models(self):
        routed = self.router.route(["math"])
        model_ids = [m.model_id for m in routed]
        self.assertNotIn("model-pending", model_ids)

    def test_routing_similarity_tag_normalization(self):
        # Case insensitive and stripped tag matching
        routed = self.router.route([" MATH ", "calculus"])
        self.assertGreater(len(routed), 0)
        self.assertEqual(routed[0].model_id, "model-math")

    def test_routing_top_k_selection_and_telemetry(self):
        routed = self.router.route(["code"], k_override=1)
        self.assertEqual(len(routed), 1)
        self.assertIn(routed[0].model_id, ["model-code", "model-general"])

        telemetry = self.router.get_telemetry()
        self.assertEqual(telemetry[routed[0].model_id], 1)

    def test_active_parameter_calculation(self):
        routed = self.router.route(["math", "code"], k_override=2)
        params = self.router.get_active_parameters(routed)
        expected = sum(m.parameter_count for m in routed)
        self.assertEqual(params, expected)

    def test_active_parameter_budget_guardrail(self):
        # Budget of 2.5M should allow model-math (1M) but prevent accumulating model-general (5M)
        budget_router = SparseTopKRouter(registry=self.registry, k=3, max_active_budget=2_500_000)
        routed = budget_router.route(["math", "general"])

        total_params = budget_router.get_active_parameters(routed)
        self.assertLessEqual(total_params, 2_500_000)

    def test_route_when_no_ready_models(self):
        empty_registry = Registry()
        empty_router = SparseTopKRouter(registry=empty_registry, k=2)
        routed = empty_router.route(["math"])
        self.assertEqual(routed, [])


if __name__ == "__main__":
    unittest.main()
