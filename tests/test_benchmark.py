import unittest

from asmodeus.benchmark import BenchmarkSuite, BenchmarkReport
from asmodeus.task_manager import TaskManager
from asmodeus.worker import WorkerAgent


class TestBenchmarkSuite(unittest.TestCase):
    def test_benchmark_execution_and_reproducibility(self):
        manager1 = TaskManager()
        worker1 = WorkerAgent(name="Worker-Node-1", skills=["coding", "python", "optimization", "math", "linear_algebra", "logic", "deduction"])
        manager1.register_worker(worker1)

        suite1 = BenchmarkSuite(seed=123)
        report1: BenchmarkReport = suite1.run(manager1)

        manager2 = TaskManager()
        worker2 = WorkerAgent(name="Worker-Node-2", skills=["coding", "python", "optimization", "math", "linear_algebra", "logic", "deduction"])
        manager2.register_worker(worker2)

        suite2 = BenchmarkSuite(seed=123)
        report2: BenchmarkReport = suite2.run(manager2)

        self.assertEqual(report1.seed, report2.seed)
        self.assertEqual(report1.total_tasks, report2.total_tasks)
        self.assertEqual(report1.successful_tasks, report2.successful_tasks)
        self.assertEqual(report1.success_rate, report2.success_rate)
        self.assertIn("coding", report1.category_scores)
        self.assertIn("math", report1.category_scores)


if __name__ == "__main__":
    unittest.main()
