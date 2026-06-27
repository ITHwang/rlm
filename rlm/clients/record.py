"""Recording LM client (RLM Lab, PRD 006) — ordered capture.

Wraps a real client and appends each call's response + usage to a per-run,
ordered :class:`Recorder` shared across the run's clients (root loop +
``llm_query`` go through one client; each ``rlm_query`` builds another). The
order is the call order; with ``max_concurrent_subcalls=1`` it is deterministic,
so :class:`~rlm.clients.replay.MockLM` can replay the list positionally.
"""

import json
import os
import threading
import time
from pathlib import Path
from typing import Any

from rlm.clients.base_lm import BaseLM
from rlm.core.types import ModelUsageSummary, UsageSummary


class Recorder:
    """Per-run ordered call log; rewritten to a fixture file as it grows."""

    def __init__(self, fixture_file: str, task_id: str | None, model_name: str):
        self.fixture_file = str(fixture_file)
        self.task_id = task_id
        self.model_name = model_name
        self.calls: list[dict[str, Any]] = []
        self._lock = threading.Lock()
        Path(self.fixture_file).parent.mkdir(parents=True, exist_ok=True)
        self._flush()

    def record(
        self,
        response: str,
        model: str,
        usage: ModelUsageSummary,
        prompt_preview: str,
        latency_s: float,
    ) -> None:
        with self._lock:
            self.calls.append(
                {
                    "response": response,
                    "model": model,
                    "usage": usage.to_dict(),
                    "latency_s": round(latency_s, 3),
                    "prompt_preview": prompt_preview,
                }
            )
            self._flush()

    def _flush(self) -> None:
        payload = {
            "task_id": self.task_id,
            "model_name": self.model_name,
            "calls": self.calls,
        }
        path = Path(self.fixture_file)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(tmp, path)


class RecordingLM(BaseLM):
    def __init__(
        self,
        model_name: str,
        recorder: Recorder,
        inner_backend: str,
        inner_kwargs: dict[str, Any] | None = None,
        sampling_args: dict[str, Any] | None = None,
        **kwargs,
    ):
        super().__init__(model_name=model_name, sampling_args=sampling_args, **kwargs)
        from rlm.clients import get_client  # local import: avoid circular import

        merged = dict(inner_kwargs or {})
        merged.setdefault("model_name", model_name)
        if sampling_args is not None:
            merged.setdefault("sampling_args", sampling_args)
        self._inner: BaseLM = get_client(inner_backend, merged)
        self.recorder = recorder

    def completion(self, prompt: str | dict[str, Any], model: str | None = None) -> str:
        start = time.perf_counter()
        response = self._inner.completion(prompt, model)
        latency = time.perf_counter() - start
        self.recorder.record(
            response, self._inner.model_name, self._inner.get_last_usage(),
            _preview(prompt), latency,
        )
        return response

    async def acompletion(self, prompt: str | dict[str, Any], model: str | None = None) -> str:
        start = time.perf_counter()
        response = await self._inner.acompletion(prompt, model)
        latency = time.perf_counter() - start
        self.recorder.record(
            response, self._inner.model_name, self._inner.get_last_usage(),
            _preview(prompt), latency,
        )
        return response

    def get_usage_summary(self) -> UsageSummary:
        return self._inner.get_usage_summary()

    def get_last_usage(self) -> ModelUsageSummary:
        return self._inner.get_last_usage()


def _preview(prompt: Any) -> str:
    if isinstance(prompt, str):
        return prompt[:200]
    if isinstance(prompt, list) and prompt:
        return str(prompt[0])[:200]
    return str(prompt)[:200]
