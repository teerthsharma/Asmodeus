import random
import time
from dataclasses import dataclass, field
from typing import List, Dict, Any, Callable

from asmodeus.task import Task
from asmodeus.worker import WorkerAgent
from asmodeus.task_manager import TaskManager


@dataclass
class BenchmarkTask:
    """Represents a deterministic benchmark query/item across coding, math, or reasoning."""
    task_id: str
    category: str
    description: str
    required_skills: List[str]
    evaluator: Callable[[Any], bool]
    should_fail_times: int = 0


@dataclass
class BenchmarkReport:
    """Summary metrics of a benchmark run."""
    seed: int
    total_tasks: int
    successful_tasks: int
    success_rate: float
    total_latency_ms: float
    category_scores: Dict[str, float]
    raw_results: List[Dict[str, Any]] = field(default_factory=list)


class BenchmarkSuite:
    """
    Deterministic benchmark suite for coding, math, and reasoning tasks.
    Ensures reproducible evaluation across runs when configured with a seed.
    """
    def __init__(self, seed: int = 42):
        self.seed = seed
        self.tasks: List[BenchmarkTask] = []
        self._build_default_task_pack()

    def _build_default_task_pack(self) -> None:
        """Populates default benchmark task pack covering core capabilities."""
        self.tasks = [
            BenchmarkTask(
                task_id="code_01",
                category="coding",
                description="Implement Fibonacci sequence generator",
                required_skills=["coding", "python"],
                evaluator=lambda res: bool(res),
            ),
            BenchmarkTask(
                task_id="code_02",
                category="coding",
                description="Refactor algorithm for O(n log n) sorting",
                required_skills=["coding", "optimization"],
                evaluator=lambda res: bool(res),
                should_fail_times=1,
            ),
            BenchmarkTask(
                task_id="math_01",
                category="math",
                description="Solve system of linear equations",
                required_skills=["math", "linear_algebra"],
                evaluator=lambda res: bool(res),
            ),
            BenchmarkTask(
                task_id="math_02",
                category="math",
                description="Compute matrix eigenvalues and trace",
                required_skills=["math", "linear_algebra"],
                evaluator=lambda res: bool(res),
            ),
            BenchmarkTask(
                task_id="reasoning_01",
                category="reasoning",
                description="Multi-step causal topology deduction",
                required_skills=["logic", "deduction"],
                evaluator=lambda res: bool(res),
            ),
        ]

    def run(self, manager: TaskManager) -> BenchmarkReport:
        """
        Runs the benchmark suite using the provided TaskManager and available workers.
        Seeds randomness for reproducible transient failure simulation and agent selection.
        """
        random.seed(self.seed)
        start_time = time.perf_counter()

        category_counts: Dict[str, int] = {}
        category_successes: Dict[str, int] = {}
        raw_results: List[Dict[str, Any]] = []

        available_workers = manager.get_available_workers()

        for btask in self.tasks:
            cat = btask.category
            category_counts[cat] = category_counts.get(cat, 0) + 1

            # Convert benchmark task to swarm Task
            task = Task(
                name=btask.description,
                should_fail_times=btask.should_fail_times,
                required_skills=btask.required_skills,
            )

            # Route task to matching workers
            matching_workers = [
                w for w in available_workers if w.can_handle_task(task)
            ]
            if not matching_workers:
                matching_workers = available_workers[:1] if available_workers else []

            assignments = {task: matching_workers}
            exec_start = time.perf_counter()
            results = manager.execute_all(assignments)
            exec_latency_ms = (time.perf_counter() - exec_start) * 1000.0

            success = results.get(task.name, False)
            if success:
                category_successes[cat] = category_successes.get(cat, 0) + 1

            raw_results.append({
                "task_id": btask.task_id,
                "category": cat,
                "description": btask.description,
                "success": success,
                "latency_ms": exec_latency_ms,
            })

        total_latency_ms = (time.perf_counter() - start_time) * 1000.0
        total_tasks = len(self.tasks)
        successful_tasks = sum(category_successes.values())
        success_rate = successful_tasks / total_tasks if total_tasks > 0 else 0.0

        category_scores = {
            cat: (category_successes.get(cat, 0) / category_counts[cat])
            for cat in category_counts
        }

        return BenchmarkReport(
            seed=self.seed,
            total_tasks=total_tasks,
            successful_tasks=successful_tasks,
            success_rate=success_rate,
            total_latency_ms=total_latency_ms,
            category_scores=category_scores,
            raw_results=raw_results,
        )
