import json
import logging
from dataclasses import dataclass, field

logger = logging.getLogger("app.registry")


@dataclass
class ToolMetrics:
    calls_total: int = 0
    denials_total: int = 0
    errors_total: int = 0
    latency_ms: list[float] = field(default_factory=list)

    def p50(self) -> float:
        return _percentile(self.latency_ms, 0.50)

    def p95(self) -> float:
        return _percentile(self.latency_ms, 0.95)


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = min(int(len(ordered) * pct), len(ordered) - 1)
    return ordered[idx]


class Metrics:
    """
    Per-tool call metrics, keyed by tool name. Kept in-process/in-memory —
    swap `record()`'s body for a Prometheus client or OTel exporter for a
    real deployment; the call sites in execution.py don't need to change.
    """

    def __init__(self) -> None:
        self._by_tool: dict[str, ToolMetrics] = {}

    def record(
        self,
        *,
        tool_name: str,
        kind: str,
        caller_id: str,
        tenant_id: str,
        scope: str,
        ok: bool,
        denied: bool,
        duration_ms: float,
    ) -> None:
        bucket = self._by_tool.setdefault(tool_name, ToolMetrics())
        bucket.calls_total += 1
        if denied:
            bucket.denials_total += 1
        elif not ok:
            bucket.errors_total += 1
        bucket.latency_ms.append(duration_ms)

        logger.info(
            json.dumps(
                {
                    "tool": tool_name,
                    "kind": kind,
                    "caller_id": caller_id,
                    "tenant_id": tenant_id,
                    "scope": scope,
                    "ok": ok,
                    "denied": denied,
                    "duration_ms": round(duration_ms, 2),
                }
            )
        )

    def summary(self) -> dict[str, dict]:
        return {
            name: {
                "calls_total": m.calls_total,
                "denials_total": m.denials_total,
                "errors_total": m.errors_total,
                "p50_latency_ms": round(m.p50(), 2),
                "p95_latency_ms": round(m.p95(), 2),
            }
            for name, m in self._by_tool.items()
        }


# Process-wide singleton, mirroring `registry`. Tests construct their own
# Metrics() instance for isolation.
metrics = Metrics()
