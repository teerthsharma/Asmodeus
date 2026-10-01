import unittest
from asmodeus.registry import Registry, ModelStatus
from asmodeus.router import SparseTopKRouter

class TestSparseTopKRouter(unittest.TestCase):
    def setUp(self):
        self.registry = Registry()
        self.registry.register_model("model_a", 1_000_000, ["python", "nlp"])
        self.registry.register_model("model_b", 10_000_000, ["python", "vision"])
        self.registry.register_model("model_c", 1_000_000, ["rust", "systems"])

        for m_id in ["model_a", "model_b", "model_c"]:
            self.registry.update_status(m_id, ModelStatus.READY)

        self.router = SparseTopKRouter(self.registry, k=2)

    def test_compute_similarity(self):
        model_a = self.registry.get_model("model_a")
        # task_tags: ['python'] (len 1), model_a capability: {'python', 'nlp'} (len 2)
        # intersection: {'python'} (len 1), union: {'python', 'nlp'} (len 2) -> sim = 0.5
        sim = self.router._compute_similarity(["python"], model_a)
        self.assertAlmostEqual(sim, 0.5)

        # No overlap -> 0.0
        sim_none = self.router._compute_similarity(["go"], model_a)
        self.assertEqual(sim_none, 0.0)

    def test_route_top_k(self):
        top_models = self.router.route(["python"])
        self.assertEqual(len(top_models), 2)
        model_ids = [m.model_id for m in top_models]
        self.assertIn("model_a", model_ids)
        self.assertIn("model_b", model_ids)

        # Telemetry should track usage
        telemetry = self.router.get_telemetry()
        self.assertEqual(telemetry.get("model_a"), 1)
        self.assertEqual(telemetry.get("model_b"), 1)

    def test_empty_ready_models(self):
        empty_registry = Registry()
        empty_router = SparseTopKRouter(empty_registry)
        results = empty_router.route(["python"])
        self.assertEqual(results, [])

if __name__ == "__main__":
    unittest.main()
