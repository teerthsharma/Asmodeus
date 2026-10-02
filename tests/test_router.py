import unittest

from asmodeus.registry import Registry, ModelStatus, MicroMaster
from asmodeus.router import SparseTopKRouter


class TestSparseTopKRouter(unittest.TestCase):
    def setUp(self):
        self.registry = Registry()
        self.router = SparseTopKRouter(self.registry, k=2)

    def test_route_empty_registry(self):
        selected = self.router.route(["coding"])
        self.assertEqual(selected, [])

    def test_route_admission_gate_only_ready_models(self):
        self.registry.register_model("model-pending", 1_000_000, ["coding"])
        self.registry.register_model("model-downloading", 1_000_000, ["coding"])
        self.registry.register_model("model-ready", 1_000_000, ["coding"])

        self.registry.update_status("model-pending", ModelStatus.PENDING)
        self.registry.update_status("model-downloading", ModelStatus.DOWNLOADING)
        self.registry.update_status("model-ready", ModelStatus.READY)

        selected = self.router.route(["coding"])
        self.assertEqual(len(selected), 1)
        self.assertEqual(selected[0].model_id, "model-ready")

    def test_topk_ranking_and_utilization_telemetry(self):
        self.registry.register_model("m1", 1_000_000, ["coding", "python"])
        self.registry.register_model("m2", 10_000_000, ["coding"])
        self.registry.register_model("m3", 1_000_000, ["math"])

        self.registry.update_status("m1", ModelStatus.READY)
        self.registry.update_status("m2", ModelStatus.READY)
        self.registry.update_status("m3", ModelStatus.READY)

        selected = self.router.route(["coding", "python"], k_override=2)
        self.assertEqual(len(selected), 2)
        # m1 has perfect capability match, so it should rank first
        self.assertEqual(selected[0].model_id, "m1")

        telemetry = self.router.get_telemetry()
        self.assertIn("m1", telemetry)
        self.assertEqual(telemetry["m1"], 1)

        # Routing again updates utilization and balances usage score
        self.router.route(["coding", "python"], k_override=2)
        self.assertEqual(self.router.get_telemetry()["m1"], 2)

    def test_compute_similarity_edge_cases(self):
        m = MicroMaster("m1", 1_000_000, capability_tags=set())
        score = self.router._compute_similarity(set(["coding"]), m)
        self.assertEqual(score, 0.0)

        score_empty_tags = self.router._compute_similarity(set(), m)
        self.assertEqual(score_empty_tags, 0.0)


if __name__ == "__main__":
    unittest.main()
