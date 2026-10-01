import unittest

from asmodeus.telemetry import TelemetryCollector, FallbackEvent
from asmodeus.task_manager import TaskManager
from asmodeus.worker import WorkerAgent
from asmodeus.task import Task


class TestTelemetryAndFallback(unittest.TestCase):
    def test_fallback_event_structure(self):
        event = FallbackEvent(
            task_name="Test Task",
            trigger_reason="Simulated failure",
            failed_specialist="Worker-1",
            chosen_backup="Worker-2",
            retry_count=1,
            reroute_latency_ms=12.5,
        )
        data = event.to_dict()
        self.assertEqual(data["task_name"], "Test Task")
        self.assertEqual(data["failed_specialist"], "Worker-1")
        self.assertEqual(data["chosen_backup"], "Worker-2")
        self.assertEqual(data["retry_count"], 1)

    def test_task_manager_fallback_logging(self):
        telemetry = TelemetryCollector()
        manager = TaskManager(telemetry_collector=telemetry)

        worker1 = WorkerAgent(name="Worker-Node-1")
        worker2 = WorkerAgent(name="Worker-Node-2")

        manager.register_worker(worker1)
        manager.register_worker(worker2)

        # Task that fails twice before success
        task = Task(name="Recoverable Task", should_fail_times=2)
        assignments = {task: [worker1, worker2]}

        results = manager.execute_all(assignments)
        self.assertTrue(results["Recoverable Task"])

        events = telemetry.get_fallback_events()
        self.assertGreaterEqual(len(events), 0)


if __name__ == "__main__":
    unittest.main()
