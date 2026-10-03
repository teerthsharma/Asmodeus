from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional

@dataclass
class FallbackEvent:
    """
    Structured record of a degraded-mode or retry fallback event.
    Captures failure triggers, affected and backup specialists, latency, and status.
    """
    task_name: str
    trigger_reason: str
    failed_specialist: str
    chosen_backup: Optional[str] = None
    retry_count: int = 0
    reroute_latency_ms: float = 0.0
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_name": self.task_name,
            "trigger_reason": self.trigger_reason,
            "failed_specialist": self.failed_specialist,
            "chosen_backup": self.chosen_backup,
            "retry_count": self.retry_count,
            "reroute_latency_ms": self.reroute_latency_ms,
            "timestamp": self.timestamp,
        }

class TelemetryCollector:
    """
    Collector for runtime telemetry including structured fallback records.
    """
    def __init__(self):
        self.fallback_events: List[FallbackEvent] = []

    def record_fallback(self, event: FallbackEvent) -> None:
        """Records a fallback event."""
        self.fallback_events.append(event)

    def get_fallback_events(self) -> List[FallbackEvent]:
        """Returns all recorded fallback events."""
        return list(self.fallback_events)

    def clear(self) -> None:
        """Clears recorded events."""
        self.fallback_events.clear()

    def get_summary_stats(self) -> Dict[str, Any]:
        """
        Computes aggregate metrics from recorded fallback events.
        """
        total_events = len(self.fallback_events)
        if total_events == 0:
            return {
                "total_fallbacks": 0,
                "total_reroute_latency_ms": 0.0,
                "average_reroute_latency_ms": 0.0,
                "failed_specialists": {},
                "trigger_reasons": {},
            }

        total_latency = sum(e.reroute_latency_ms for e in self.fallback_events)
        avg_latency = total_latency / total_events

        failed_specialists: Dict[str, int] = {}
        trigger_reasons: Dict[str, int] = {}

        for event in self.fallback_events:
            failed_specialists[event.failed_specialist] = (
                failed_specialists.get(event.failed_specialist, 0) + 1
            )
            trigger_reasons[event.trigger_reason] = (
                trigger_reasons.get(event.trigger_reason, 0) + 1
            )

        return {
            "total_fallbacks": total_events,
            "total_reroute_latency_ms": total_latency,
            "average_reroute_latency_ms": avg_latency,
            "failed_specialists": failed_specialists,
            "trigger_reasons": trigger_reasons,
        }
