"""CPython-on-Wasm REPL environment for RLM Lab PRD-009.

This environment embeds upstream ``pyeryx`` (imported as ``eryx``) and adapts
RLM's synchronous Python REPL contract to eryx's async host-callback model.
"""

from __future__ import annotations

import ast
import copy
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

from rlm.core.comms_utils import LMRequest, send_lm_request, send_lm_request_batched
from rlm.core.types import REPLResult, RLMChatCompletion
from rlm.environments.base_env import NonIsolatedEnv

_ASYNC_RESERVED_CALLS = {
    "llm_query",
    "llm_query_batched",
    "rlm_query",
    "rlm_query_batched",
}
_FIRST_ARG_NAME = {
    "llm_query": "prompt",
    "llm_query_batched": "prompts",
    "rlm_query": "prompt",
    "rlm_query_batched": "prompts",
}


class _AsyncReservedCallRewriter(ast.NodeTransformer):
    """Wrap reserved host-callback calls in ``await`` for pyeryx.

    RLM prompts and recorded trajectories call ``llm_query(...)`` synchronously.
    Eryx exposes callbacks as async guest functions, so the code sent to eryx
    must use ``await llm_query(...)``. This transformer preserves the prompt
    contract while adapting the execution substrate.
    """

    def __init__(self) -> None:
        self._inside_await = 0

    def visit_Await(self, node: ast.Await) -> ast.AST:
        self._inside_await += 1
        try:
            return self.generic_visit(node)
        finally:
            self._inside_await -= 1

    def visit_Call(self, node: ast.Call) -> ast.AST:
        node = self.generic_visit(node)
        if not isinstance(node.func, ast.Name) or node.func.id not in _ASYNC_RESERVED_CALLS:
            return node
        node = _keywordize_reserved_call(node)
        if self._inside_await:
            return node
        return ast.copy_location(ast.Await(value=node), node)


def _keywordize_reserved_call(node: ast.Call) -> ast.Call:
    """Convert positional reserved-callback args to pyeryx callback kwargs."""

    name = node.func.id
    args = list(node.args)
    keywords = list(node.keywords)
    if args:
        keywords.insert(0, ast.keyword(arg=_FIRST_ARG_NAME[name], value=args.pop(0)))
    if args:
        keywords.insert(1, ast.keyword(arg="model", value=args.pop(0)))
    node.args = args
    node.keywords = keywords
    return node


def rewrite_reserved_callback_calls(code: str) -> str:
    """Return code with RLM reserved callback calls adapted for pyeryx."""

    tree = ast.parse(code)
    tree = _AsyncReservedCallRewriter().visit(tree)
    ast.fix_missing_locations(tree)
    return ast.unparse(tree)


class PyWasmEnv(NonIsolatedEnv):
    """RLM environment backed by a persistent pyeryx Session."""

    def __init__(
        self,
        lm_handler_address: tuple[str, int] | None = None,
        context_payload: dict | list | str | None = None,
        persistent: bool = False,
        depth: int = 1,
        subcall_fn=None,
        max_concurrent_subcalls: int = 4,
        execution_timeout_ms: int = 600_000,
        max_fuel: int | None = None,
        deterministic_seed: int | None = None,
        **kwargs,
    ):
        super().__init__(
            persistent=persistent,
            depth=depth,
            max_concurrent_subcalls=max_concurrent_subcalls,
            **kwargs,
        )
        self.lm_handler_address = lm_handler_address
        self.subcall_fn = subcall_fn
        self.execution_timeout_ms = execution_timeout_ms
        self.max_fuel = max_fuel
        self.deterministic_seed = deterministic_seed
        self._lock = threading.Lock()
        self._timing_lock = threading.Lock()
        self._pending_llm_calls: list[RLMChatCompletion] = []
        self._context_count = 0
        self._history_count = 0
        self._shadow_locals: dict[str, Any] = {}
        self._last_stats: dict[str, Any] = {}
        self._timing = self._new_timing()
        self._last_execute_ended_at: float | None = None
        self._inter_turn_idle_started_at: float | None = None
        self.session = None
        self.eryx = None
        self.setup()
        if context_payload is not None:
            self.load_context(context_payload)

    def setup(self) -> None:
        try:
            import eryx
        except ImportError as exc:
            raise RuntimeError(
                "PyWasmEnv requires pyeryx. Install the backend dependencies "
                "or run with `uv run --with pyeryx ...`."
            ) from exc

        self.eryx = eryx
        self.session = eryx.Session(
            result_variable="answer",
            execution_timeout_ms=self.execution_timeout_ms,
            max_fuel=self.max_fuel,
            callbacks=[
                {
                    "name": "__rlm_host_llm_query",
                    "fn": self._llm_query,
                    "description": "Single LM completion",
                },
                {
                    "name": "__rlm_host_llm_query_batched",
                    "fn": self._llm_query_batched,
                    "description": "Batched LM completions",
                },
                {
                    "name": "__rlm_host_rlm_query",
                    "fn": self._rlm_query,
                    "description": "Recursive RLM completion",
                },
                {
                    "name": "__rlm_host_rlm_query_batched",
                    "fn": self._rlm_query_batched,
                    "description": "Batched recursive RLM completions",
                },
            ],
        )
        self._pending_llm_calls = []
        self._shadow_locals = {}
        self._install_scaffold()

    def load_context(self, context_payload: dict | list | str):
        self.add_context(context_payload, 0)

    def add_context(
        self, context_payload: dict | list | str, context_index: int | None = None
    ) -> int:
        if context_index is None:
            context_index = self._context_count
        var_name = f"context_{context_index}"
        self._set_json_global(var_name, context_payload)
        self._shadow_locals[var_name] = copy.deepcopy(context_payload)
        if context_index == 0:
            self._execute_setup("context = context_0")
            self._shadow_locals["context"] = self._shadow_locals[var_name]
        self._context_count = max(self._context_count, context_index + 1)
        return context_index

    def update_handler_address(self, address: tuple[str, int]) -> None:
        self.lm_handler_address = address

    def get_context_count(self) -> int:
        return self._context_count

    def add_history(
        self, message_history: list[dict[str, Any]], history_index: int | None = None
    ) -> int:
        if history_index is None:
            history_index = self._history_count
        var_name = f"history_{history_index}"
        payload = copy.deepcopy(message_history)
        self._set_json_global(var_name, payload)
        self._shadow_locals[var_name] = payload
        if history_index == 0:
            self._execute_setup("history = history_0")
            self._shadow_locals["history"] = self._shadow_locals[var_name]
        self._history_count = max(self._history_count, history_index + 1)
        return history_index

    def get_history_count(self) -> int:
        return self._history_count

    def reset_for_request(self) -> None:
        """Clear request-scoped session state while keeping the pyeryx object warm."""

        if self.session is not None:
            self.session.clear_state()
        self._context_count = 0
        self._history_count = 0
        self._shadow_locals = {}
        self._pending_llm_calls = []
        self._last_stats = {}
        self._timing = self._new_timing()
        self._last_execute_ended_at = None
        self._inter_turn_idle_started_at = None
        self._install_scaffold()

    def execute_code(self, code: str) -> REPLResult:
        if self.session is None:
            raise RuntimeError("PyWasmEnv is not initialized")

        start_time = time.perf_counter()
        park_before = self._timing_value("mid_execute_park_seconds")
        end_time = start_time
        with self._lock:
            self._pending_llm_calls = []
            self._install_scaffold()
            wrapped = self._wrap_user_code(code)
            try:
                result = self.session.execute(wrapped)
                end_time = time.perf_counter()
                self._record_execute_timing(start_time, end_time, park_before)
                stdout = result.stdout
                stderr = result.stderr
                answer = result.result if isinstance(result.result, dict) else None
                final_answer = (
                    str(answer.get("content", ""))
                    if answer is not None and answer.get("ready")
                    else None
                )
                self._last_stats = {
                    "duration_ms": result.duration_ms,
                    "callback_invocations": result.callback_invocations,
                    "peak_memory_bytes": result.peak_memory_bytes,
                    "fuel_consumed": result.fuel_consumed,
                    "result_error": result.result_error,
                    "idle_timing": self.runtime_timing(),
                }
                if result.result_error:
                    stderr = (stderr + "\n" if stderr else "") + result.result_error
                locals_snapshot = {
                    **self._shadow_locals,
                    "answer": answer or {"ready": False, "content": ""},
                    "__pywasm_stats": dict(self._last_stats),
                }
            except Exception as exc:  # noqa: BLE001 - match LocalREPL behavior.
                end_time = time.perf_counter()
                self._record_execute_timing(start_time, end_time, park_before)
                stdout = ""
                stderr = f"\n{type(exc).__name__}: {exc}"
                final_answer = None
                self._last_stats = {
                    "duration_ms": None,
                    "callback_invocations": None,
                    "peak_memory_bytes": None,
                    "fuel_consumed": None,
                    "result_error": f"{type(exc).__name__}: {exc}",
                    "idle_timing": self.runtime_timing(),
                }
                locals_snapshot = {
                    **self._shadow_locals,
                    "answer": {"ready": False, "content": ""},
                    "__pywasm_stats": dict(self._last_stats),
                }
        return REPLResult(
            stdout=stdout,
            stderr=stderr,
            locals=locals_snapshot,
            execution_time=end_time - start_time,
            rlm_calls=self._pending_llm_calls.copy(),
            final_answer=final_answer,
        )

    def cleanup(self) -> None:
        self.session = None
        self._shadow_locals.clear()
        self._pending_llm_calls.clear()

    def begin_inter_turn_idle(self) -> None:
        """Mark that the session is awaiting the next root-model response."""

        with self._timing_lock:
            if (
                self._last_execute_ended_at is not None
                and self._inter_turn_idle_started_at is None
            ):
                self._inter_turn_idle_started_at = self._last_execute_ended_at

    def end_inter_turn_idle(self) -> None:
        with self._timing_lock:
            if self._inter_turn_idle_started_at is None:
                return
            now = time.perf_counter()
            self._timing["inter_turn_idle_seconds"] += max(
                0.0,
                now - self._inter_turn_idle_started_at,
            )
            self._timing["inter_turn_idle_count"] += 1
            self._inter_turn_idle_started_at = None
            self._last_execute_ended_at = None

    def runtime_timing(self) -> dict[str, Any]:
        with self._timing_lock:
            timing = dict(self._timing)
            if self._inter_turn_idle_started_at is not None:
                timing["inter_turn_idle_seconds"] += max(
                    0.0,
                    time.perf_counter() - self._inter_turn_idle_started_at,
                )
        active = float(timing["active_seconds"])
        inter_turn = float(timing["inter_turn_idle_seconds"])
        mid_execute = float(timing["mid_execute_park_seconds"])
        observed = active + inter_turn + mid_execute
        timing["observed_seconds"] = observed
        timing["active_fraction"] = active / observed if observed else 0.0
        timing["inter_turn_idle_fraction"] = (
            inter_turn / observed if observed else 0.0
        )
        timing["mid_execute_park_fraction"] = (
            mid_execute / observed if observed else 0.0
        )
        return timing

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup()
        return False

    def _execute_setup(self, code: str) -> None:
        if self.session is None:
            raise RuntimeError("PyWasmEnv is not initialized")
        self.session.execute(code)

    def _install_scaffold(self) -> None:
        self._execute_setup(
            """
async def llm_query(prompt, model=None):
    return await __rlm_host_llm_query(prompt=prompt, model=model)

async def llm_query_batched(prompts, model=None):
    return await __rlm_host_llm_query_batched(prompts=prompts, model=model)

async def rlm_query(prompt, model=None):
    return await __rlm_host_rlm_query(prompt=prompt, model=model)

async def rlm_query_batched(prompts, model=None):
    return await __rlm_host_rlm_query_batched(prompts=prompts, model=model)

def SHOW_VARS():
    names = sorted(
        name for name in globals()
        if not name.startswith("_") and name not in {"SHOW_VARS", "answer"}
    )
    return "Available variables: " + repr(names)

if "context_0" in globals():
    context = context_0
if "history_0" in globals():
    history = history_0
"""
        )

    def _set_json_global(self, name: str, payload: Any) -> None:
        json_text = json.dumps(payload)
        self._execute_setup(
            "import json\n" f"{name} = json.loads({json.dumps(json_text)})"
        )

    def _wrap_user_code(self, code: str) -> str:
        rewritten = rewrite_reserved_callback_calls(code)
        prefix = 'answer = {"ready": False, "content": ""}\n'
        if self.deterministic_seed is not None:
            seed = int(self.deterministic_seed)
            prefix += (
                "import random as _rlm_random\n"
                f"_rlm_random.seed({seed})\n"
                "try:\n"
                "    import numpy as _rlm_numpy\n"
                f"    _rlm_numpy.random.seed({seed})\n"
                "except Exception:\n"
                "    pass\n"
            )
        return prefix + rewritten

    def _llm_query(self, prompt: str, model: str | None = None) -> str:
        park_start = time.perf_counter()
        try:
            if not self.lm_handler_address:
                return "Error: No LM handler configured"
            try:
                request = LMRequest(prompt=prompt, model=model, depth=self.depth)
                response = send_lm_request(self.lm_handler_address, request)
                if not response.success:
                    return f"Error: {response.error}"
                self._pending_llm_calls.append(response.chat_completion)
                return response.chat_completion.response
            except Exception as exc:  # noqa: BLE001 - returned to guest as data.
                return f"Error: LM query failed - {exc}"
        finally:
            self._record_mid_execute_park(park_start)

    def _llm_query_batched(
        self, prompts: list[str], model: str | None = None
    ) -> list[str]:
        park_start = time.perf_counter()
        try:
            if not self.lm_handler_address:
                return ["Error: No LM handler configured"] * len(prompts)
            try:
                responses = send_lm_request_batched(
                    self.lm_handler_address, prompts, model=model, depth=self.depth
                )
                results = []
                for response in responses:
                    if not response.success:
                        results.append(f"Error: {response.error}")
                    else:
                        self._pending_llm_calls.append(response.chat_completion)
                        results.append(response.chat_completion.response)
                return results
            except Exception as exc:  # noqa: BLE001
                return [f"Error: LM query failed - {exc}"] * len(prompts)
        finally:
            self._record_mid_execute_park(park_start)

    def _rlm_query(self, prompt: str, model: str | None = None) -> str:
        if self.subcall_fn is None:
            return self._llm_query(prompt, model)
        park_start = time.perf_counter()
        try:
            completion = self.subcall_fn(prompt, model)
            self._pending_llm_calls.append(completion)
            return completion.response
        except Exception as exc:  # noqa: BLE001
            return f"Error: RLM query failed - {exc}"
        finally:
            self._record_mid_execute_park(park_start)

    def _rlm_query_batched(
        self, prompts: list[str], model: str | None = None
    ) -> list[str]:
        if self.subcall_fn is not None:
            if len(prompts) <= 1:
                return [self._rlm_query(prompt, model) for prompt in prompts]

            park_start = time.perf_counter()
            try:
                max_workers = min(self.max_concurrent_subcalls, len(prompts))
                results: list[str] = [""] * len(prompts)
                completions: list[tuple[int, RLMChatCompletion]] = []
                lock = threading.Lock()

                def _run_subcall(index: int, prompt: str) -> None:
                    try:
                        completion = self.subcall_fn(prompt, model)
                        with lock:
                            completions.append((index, completion))
                        results[index] = completion.response
                    except Exception as exc:  # noqa: BLE001
                        results[index] = f"Error: RLM query failed - {exc}"

                with ThreadPoolExecutor(max_workers=max_workers) as executor:
                    futures = [
                        executor.submit(_run_subcall, i, prompt)
                        for i, prompt in enumerate(prompts)
                    ]
                    for future in as_completed(futures):
                        future.result()

                completions.sort(key=lambda x: x[0])
                for _, completion in completions:
                    self._pending_llm_calls.append(completion)
                return results
            finally:
                self._record_mid_execute_park(park_start)
        return self._llm_query_batched(prompts, model)

    @staticmethod
    def _new_timing() -> dict[str, float | int]:
        return {
            "execute_wall_seconds": 0.0,
            "active_seconds": 0.0,
            "mid_execute_park_seconds": 0.0,
            "inter_turn_idle_seconds": 0.0,
            "execute_count": 0,
            "mid_execute_park_count": 0,
            "inter_turn_idle_count": 0,
        }

    def _timing_value(self, key: str) -> float:
        with self._timing_lock:
            return float(self._timing[key])

    def _record_execute_timing(
        self,
        started_at: float,
        ended_at: float,
        park_before: float,
    ) -> None:
        elapsed = max(0.0, ended_at - started_at)
        with self._timing_lock:
            park_delta = max(
                0.0,
                float(self._timing["mid_execute_park_seconds"]) - park_before,
            )
            self._timing["execute_wall_seconds"] += elapsed
            self._timing["active_seconds"] += max(0.0, elapsed - park_delta)
            self._timing["execute_count"] += 1
            self._last_execute_ended_at = ended_at

    def _record_mid_execute_park(self, started_at: float) -> None:
        elapsed = max(0.0, time.perf_counter() - started_at)
        with self._timing_lock:
            self._timing["mid_execute_park_seconds"] += elapsed
            self._timing["mid_execute_park_count"] += 1
