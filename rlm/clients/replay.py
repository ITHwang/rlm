"""Replay LM client (RLM Lab, PRD 006) — ordered (positional) replay.

Returns recorded responses by CALL ORDER, not by prompt content. Keying on the
prompt is brittle: the prompt is built from the previous step's output, so any
run-to-run noise in that output (timestamps, memory addresses, temp-dir uuids,
network results) changes the key and breaks the lookup. Call order depends only
on the code's control flow, which is stable — so ordered replay is immune to that
cosmetic noise.

A per-run :class:`ReplaySession` holds the ordered responses + a cursor, shared
across every client the RLM builds for one run (root loop + ``llm_query`` go
through one client; each ``rlm_query`` sub-call builds another). The session is
passed via ``backend_kwargs`` (which ``_subcall`` shallow-copies, preserving the
reference), so all of a run's clients advance the same cursor. Concurrent load
requests each get their own session, so they never interfere. Deterministic order
requires sequential sub-calls (``max_concurrent_subcalls=1``).
"""

import json
import threading
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

from rlm.clients.base_lm import BaseLM
from rlm.core.types import ModelUsageSummary, UsageSummary


class MockLMMiss(KeyError):
    """Raised when replay needs more calls than were recorded (divergence)."""


class ReplaySession:
    """Per-run ordered cursor over recorded calls, shared across a run's clients."""

    def __init__(self, calls: list[dict[str, Any]], replay_latency: bool = False):
        self.calls = calls
        self.idx = 0
        self.had_miss = False
        # When True, sleep each call's recorded model latency so a replayed
        # session stays alive ~as long as in production (faithful lifecycle /
        # throughput). Off for verification/recovery (just checking reproduction).
        self.replay_latency = replay_latency
        self._lock = threading.Lock()
        self._model_calls: dict[str, int] = defaultdict(int)
        self._model_in: dict[str, int] = defaultdict(int)
        self._model_out: dict[str, int] = defaultdict(int)
        self._model_cost: dict[str, float] = defaultdict(float)
        self.last_usage = ModelUsageSummary(0, 0, 0, None)

    @classmethod
    def from_file(cls, fixture_file: str, replay_latency: bool = False) -> "ReplaySession":
        return cls(load_calls(fixture_file), replay_latency=replay_latency)

    def next(self, default_model: str) -> str:
        with self._lock:
            if self.idx >= len(self.calls):
                self.had_miss = True
                raise MockLMMiss(
                    f"Replay needs call #{self.idx + 1} but only "
                    f"{len(self.calls)} were recorded (control-flow divergence)."
                )
            entry = self.calls[self.idx]
            self.idx += 1
            model = entry.get("model") or default_model
            if self.replay_latency and entry.get("latency_s"):
                time.sleep(float(entry["latency_s"]))
            usage = ModelUsageSummary.from_dict(entry.get("usage") or {})
            self._model_calls[model] += 1
            self._model_in[model] += usage.total_input_tokens or 0
            self._model_out[model] += usage.total_output_tokens or 0
            if usage.total_cost:
                self._model_cost[model] += usage.total_cost
            self.last_usage = usage
            return entry["response"]

    def consumed_all(self) -> bool:
        return self.idx == len(self.calls)

    def usage_summary(self) -> UsageSummary:
        return UsageSummary(
            model_usage_summaries={
                model: ModelUsageSummary(
                    total_calls=self._model_calls[model],
                    total_input_tokens=self._model_in[model],
                    total_output_tokens=self._model_out[model],
                    total_cost=self._model_cost[model] or None,
                )
                for model in self._model_calls
            }
        )


class MockLM(BaseLM):
    """Thin ``BaseLM`` over a :class:`ReplaySession`; returns responses in order."""

    def __init__(
        self,
        model_name: str,
        session: ReplaySession,
        sampling_args: dict[str, Any] | None = None,
        **kwargs,
    ):
        super().__init__(model_name=model_name, sampling_args=sampling_args, **kwargs)
        self.session = session

    def completion(self, prompt: str | dict[str, Any], model: str | None = None) -> str:
        return self.session.next(self.model_name)

    async def acompletion(self, prompt: str | dict[str, Any], model: str | None = None) -> str:
        return self.session.next(self.model_name)

    def get_usage_summary(self) -> UsageSummary:
        return self.session.usage_summary()

    def get_last_usage(self) -> ModelUsageSummary:
        return self.session.last_usage


def load_calls(fixture_file: str) -> list[dict[str, Any]]:
    path = Path(fixture_file)
    if not path.exists():
        raise FileNotFoundError(f"Replay fixture not found: {fixture_file}")
    data = json.loads(path.read_text(encoding="utf-8"))
    calls = data.get("calls") if isinstance(data, dict) else None
    if not isinstance(calls, list):
        raise ValueError(f"Fixture {fixture_file} has no 'calls' list.")
    return calls
