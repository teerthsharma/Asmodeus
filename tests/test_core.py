import unittest
import logging
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from asmodeus.scout import ScoutAgent
from asmodeus.worker import WorkerAgent
from asmodeus.task_manager import TaskManager
from asmodeus.task import Task

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

class TestAsmodeusSystem(unittest.TestCase):
    def setUp(self):
        self.scout = ScoutAgent(name="Scout-Alpha")
        self.worker1 = WorkerAgent(name="Worker-Node-1")
        self.worker2 = WorkerAgent(name="Worker-Node-2")
        self.manager = TaskManager()
        
        self.manager.register_worker(self.worker1)
        self.manager.register_worker(self.worker2)

    def test_end_to_end_execution_and_recovery(self):
        assignments = self.manager.assign_tasks_from_scout(self.scout)
        self.assertEqual(len(assignments), 3)
        
        results = self.manager.execute_all(assignments)
        
        self.assertTrue(results["Initial System Recon"])
        self.assertTrue(results["Data Gathering"])
        self.assertTrue(results["Final Topology Report"])

    def test_worker_permanent_failure(self):
        impossible_task = Task(name="Impossible Calculation", should_fail_times=5)
        success = self.worker1.execute_task(impossible_task)
        
        self.assertFalse(success)
        self.assertEqual(impossible_task.failures, 4)

    def test_scout_task_routing(self):
        task = Task(name="Simple Task")
        assigned = self.scout.route_task_to_agents(task, self.manager.workers)
        
        self.assertEqual(len(assigned), 1)
        self.assertEqual(assigned[0].name, "Worker-Node-1")

    def test_subagent_skill_assignment_and_completion(self):
        specialist = self.manager.create_subagent(
            parent_worker_name="Worker-Node-1",
            subagent_name="Worker-Node-1-Network",
            skills=["networking", "recon"],
        )

        skill_task = Task(name="Network Recon", required_skills=["networking"], should_fail_times=1)

        routed = self.scout.route_task_to_agents(skill_task, self.manager.get_available_workers())
        self.assertGreaterEqual(len(routed), 1)
        self.assertEqual(routed[0].name, specialist.name)

        results = self.manager.execute_all({skill_task: routed})
        self.assertTrue(results["Network Recon"])
        self.assertTrue(skill_task.executed)

if __name__ == '__main__':
    unittest.main()
