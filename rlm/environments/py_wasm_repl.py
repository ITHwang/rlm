"""CPython-on-Wasm REPL environment for RLM Lab PRD-009.

This environment embeds upstream ``pyeryx`` (imported as ``eryx``) and adapts
RLM's synchronous Python REPL contract to eryx's async host-callback model.
"""

from __future__ import annotations

import ast
import copy
import gc
import json
import os
import tempfile
import textwrap
import threading
import time
import uuid
from collections import deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
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


def _current_process_rss_bytes() -> int | None:
    try:
        import psutil

        return int(psutil.Process().memory_info().rss)
    except Exception:  # noqa: BLE001 - RSS is diagnostic only.
        pass

    try:
        with open("/proc/self/statm", encoding="utf-8") as handle:
            fields = handle.read().split()
        if len(fields) < 2:
            return None
        return int(fields[1]) * int(os.sysconf("SC_PAGE_SIZE"))
    except Exception:  # noqa: BLE001 - macOS has no /proc/self/statm.
        return None


@dataclass
class _TieredMemoryEntry:
    session_id: str
    compressed: bytes | None
    tier: str
    created_at: float
    last_access_at: float
    disk_path: Path | None
    snapshot_bytes: int
    compressed_bytes: int
    regions_before_demote: list[dict[str, Any]]
    snapshot_ms: float
    compression_ms: float
    rss_before_demote_bytes: int | None
    rss_after_snapshot_bytes: int | None
    rss_after_compress_bytes: int | None
    rss_after_drop_live_bytes: int | None


class _TieredMemoryStore:
    """Process-local Tier 2/Tier 3 store shared by in-process PyWasm workers."""

    def __init__(
        self,
        *,
        tier2_budget_bytes: int,
        spill_dir: Path,
        ttl_seconds: float | None,
    ) -> None:
        self.tier2_budget_bytes = max(0, int(tier2_budget_bytes))
        self.spill_dir = spill_dir
        self.ttl_seconds = ttl_seconds
        self._lock = threading.Lock()
        self._entries: dict[str, _TieredMemoryEntry] = {}
        self._live_session_ids: set[str] = set()
        self._tier2_bytes = 0
        self._tier3_bytes = 0
        self._demote_count = 0
        self._promote_count = 0
        self._spill_count = 0
        self._discard_count = 0
        self._ttl_cleanup_count = 0
        self._demote_latencies_ms: deque[float] = deque(maxlen=2000)
        self._promote_latencies_ms: deque[float] = deque(maxlen=2000)

    def mark_live(self, session_id: str) -> None:
        with self._lock:
            self._live_session_ids.add(session_id)

    def unregister(self, session_id: str) -> None:
        with self._lock:
            self._live_session_ids.discard(session_id)
            self._discard_locked(session_id)

    def discard(self, session_id: str) -> None:
        with self._lock:
            self._discard_locked(session_id)

    def put(self, entry: _TieredMemoryEntry) -> str:
        with self._lock:
            self._cleanup_expired_locked(time.monotonic())
            self._discard_locked(entry.session_id)
            self._live_session_ids.discard(entry.session_id)
            self._entries[entry.session_id] = entry
            self._tier2_bytes += entry.compressed_bytes
            self._demote_count += 1
            self._demote_latencies_ms.append(entry.snapshot_ms + entry.compression_ms)
            self._enforce_budget_locked()
            return self._entries[entry.session_id].tier

    def record_promote_latency_ms(self, latency_ms: float) -> None:
        with self._lock:
            self._promote_latencies_ms.append(latency_ms)

    def take(self, session_id: str) -> tuple[_TieredMemoryEntry, bytes] | None:
        with self._lock:
            self._cleanup_expired_locked(time.monotonic())
            entry = self._entries.pop(session_id, None)
            if entry is None:
                return None
            if entry.tier == "tier2":
                compressed = entry.compressed
                if compressed is None:
                    raise RuntimeError("Tier 2 entry is missing compressed bytes")
                self._tier2_bytes -= entry.compressed_bytes
            elif entry.tier == "tier3":
                if entry.disk_path is None:
                    raise RuntimeError("Tier 3 entry is missing disk path")
                compressed = entry.disk_path.read_bytes()
                self._tier3_bytes -= entry.compressed_bytes
                with contextlib_suppress_file_errors():
                    entry.disk_path.unlink()
            else:
                raise RuntimeError(f"Unknown tiered-memory tier: {entry.tier}")
            self._promote_count += 1
            self._live_session_ids.add(session_id)
            return entry, compressed

    def tier_for(self, session_id: str) -> str | None:
        with self._lock:
            if session_id in self._live_session_ids:
                return "tier1"
            entry = self._entries.get(session_id)
            return entry.tier if entry is not None else None

    def stats(self) -> dict[str, Any]:
        with self._lock:
            self._cleanup_expired_locked(time.monotonic())
            tier2_entries = sum(
                1 for entry in self._entries.values() if entry.tier == "tier2"
            )
            tier3_entries = sum(
                1 for entry in self._entries.values() if entry.tier == "tier3"
            )
            return {
                "tier1_live_sessions": len(self._live_session_ids),
                "tier2_entries": tier2_entries,
                "tier2_bytes": self._tier2_bytes,
                "tier2_budget_bytes": self.tier2_budget_bytes,
                "tier3_entries": tier3_entries,
                "tier3_bytes": self._tier3_bytes,
                "demote_count": self._demote_count,
                "promote_count": self._promote_count,
                "spill_count": self._spill_count,
                "discard_count": self._discard_count,
                "ttl_cleanup_count": self._ttl_cleanup_count,
                "demote_latency_ms_p50": _percentile(
                    sorted(self._demote_latencies_ms), 0.50
                ),
                "demote_latency_ms_max": (
                    max(self._demote_latencies_ms)
                    if self._demote_latencies_ms
                    else None
                ),
                "promote_latency_ms_p50": _percentile(
                    sorted(self._promote_latencies_ms), 0.50
                ),
                "promote_latency_ms_max": (
                    max(self._promote_latencies_ms)
                    if self._promote_latencies_ms
                    else None
                ),
            }

    def _enforce_budget_locked(self) -> None:
        while self._tier2_bytes > self.tier2_budget_bytes:
            candidates = [
                entry for entry in self._entries.values() if entry.tier == "tier2"
            ]
            if not candidates:
                return
            victim = min(candidates, key=lambda entry: entry.last_access_at)
            self._spill_locked(victim)

    def _spill_locked(self, entry: _TieredMemoryEntry) -> None:
        if entry.compressed is None:
            return
        self.spill_dir.mkdir(parents=True, exist_ok=True)
        path = self.spill_dir / f"{entry.session_id}-{int(entry.created_at * 1_000_000)}.zst"
        path.write_bytes(entry.compressed)
        entry.compressed = None
        entry.disk_path = path
        entry.tier = "tier3"
        self._tier2_bytes -= entry.compressed_bytes
        self._tier3_bytes += entry.compressed_bytes
        self._spill_count += 1

    def _discard_locked(self, session_id: str) -> None:
        entry = self._entries.pop(session_id, None)
        if entry is None:
            return
        if entry.tier == "tier2":
            self._tier2_bytes -= entry.compressed_bytes
        elif entry.tier == "tier3":
            self._tier3_bytes -= entry.compressed_bytes
            if entry.disk_path is not None:
                with contextlib_suppress_file_errors():
                    entry.disk_path.unlink()
        self._discard_count += 1

    def _cleanup_expired_locked(self, now: float) -> None:
        if self.ttl_seconds is None:
            return
        expired = [
            session_id
            for session_id, entry in self._entries.items()
            if now - entry.created_at >= self.ttl_seconds
        ]
        for session_id in expired:
            self._discard_locked(session_id)
            self._ttl_cleanup_count += 1


class contextlib_suppress_file_errors:
    def __enter__(self):
        return None

    def __exit__(self, exc_type, exc_val, exc_tb):
        return exc_type is not None and issubclass(exc_type, OSError)


_TIERED_MEMORY_STORES: dict[str, _TieredMemoryStore] = {}
_TIERED_MEMORY_STORES_LOCK = threading.Lock()


def py_wasm_tiered_memory_global_stats() -> dict[str, Any]:
    """Return aggregate process-local tiered-memory stats for Prometheus."""

    with _TIERED_MEMORY_STORES_LOCK:
        stores = list(_TIERED_MEMORY_STORES.values())
    totals: dict[str, Any] = {
        "tier1_live_sessions": 0,
        "tier2_entries": 0,
        "tier2_bytes": 0,
        "tier2_budget_bytes": 0,
        "tier3_entries": 0,
        "tier3_bytes": 0,
        "demote_count": 0,
        "promote_count": 0,
        "spill_count": 0,
        "discard_count": 0,
        "ttl_cleanup_count": 0,
        "demote_latency_ms_p50": None,
        "demote_latency_ms_max": None,
        "promote_latency_ms_p50": None,
        "promote_latency_ms_max": None,
    }
    for store in stores:
        stats = store.stats()
        for key in (
            "tier1_live_sessions",
            "tier2_entries",
            "tier2_bytes",
            "tier2_budget_bytes",
            "tier3_entries",
            "tier3_bytes",
            "demote_count",
            "promote_count",
            "spill_count",
            "discard_count",
            "ttl_cleanup_count",
        ):
            totals[key] += stats.get(key, 0) or 0
        for key in (
            "demote_latency_ms_p50",
            "demote_latency_ms_max",
            "promote_latency_ms_p50",
            "promote_latency_ms_max",
        ):
            value = stats.get(key)
            if value is not None:
                current = totals[key]
                totals[key] = value if current is None else max(current, value)
    totals["store_count"] = len(stores)
    return totals


def _get_tiered_memory_store(config: dict[str, Any]) -> _TieredMemoryStore:
    spill_dir = Path(config["spill_dir"])
    key = "|".join(
        [
            str(spill_dir),
            str(config["tier2_budget_bytes"]),
            str(config.get("ttl_seconds")),
        ]
    )
    with _TIERED_MEMORY_STORES_LOCK:
        store = _TIERED_MEMORY_STORES.get(key)
        if store is None:
            store = _TieredMemoryStore(
                tier2_budget_bytes=int(config["tier2_budget_bytes"]),
                spill_dir=spill_dir,
                ttl_seconds=config.get("ttl_seconds"),
            )
            _TIERED_MEMORY_STORES[key] = store
        return store


def _encode_linear_memory_regions(regions: list[dict[str, Any]]) -> bytes:
    metadata: list[dict[str, Any]] = []
    chunks: list[bytes] = []
    for region in regions:
        region_bytes = bytes(region["bytes"])
        metadata.append(
            {
                **{key: value for key, value in region.items() if key != "bytes"},
                "encoded_byte_len": len(region_bytes),
            }
        )
        chunks.append(region_bytes)
    header = json.dumps(metadata, separators=(",", ":"), sort_keys=True).encode()
    return len(header).to_bytes(8, "big") + header + b"".join(chunks)


def _decode_linear_memory_regions(blob: bytes) -> list[dict[str, Any]]:
    header_len = int.from_bytes(blob[:8], "big")
    header_end = 8 + header_len
    metadata = json.loads(blob[8:header_end].decode())
    offset = header_end
    regions: list[dict[str, Any]] = []
    for item in metadata:
        byte_len = int(item.pop("encoded_byte_len"))
        regions.append({**item, "bytes": blob[offset : offset + byte_len]})
        offset += byte_len
    if offset != len(blob):
        raise RuntimeError("Decoded tiered-memory snapshot has trailing bytes")
    return regions


def _parse_memory_limit_bytes(value: str | None) -> int | None:
    if not value:
        return None
    text = value.strip().lower()
    try:
        if text.endswith("g"):
            return int(float(text[:-1]) * 1024**3)
        if text.endswith("m"):
            return int(float(text[:-1]) * 1024**2)
        if text.endswith("k"):
            return int(float(text[:-1]) * 1024)
        return int(text)
    except ValueError:
        return None


def _percentile(sorted_values: list[float], q: float) -> float | None:
    if not sorted_values:
        return None
    if len(sorted_values) == 1:
        return sorted_values[0]
    pos = q * (len(sorted_values) - 1)
    lo = int(pos)
    hi = min(lo + 1, len(sorted_values) - 1)
    frac = pos - lo
    return sorted_values[lo] * (1 - frac) + sorted_values[hi] * frac


def _normalize_tiered_memory_config(config: dict[str, Any] | None) -> dict[str, Any] | None:
    if not config:
        return None
    policy = str(config.get("policy", "graceful_idle_lru"))
    if policy != "graceful_idle_lru":
        raise ValueError(f"Unsupported tiered_memory.policy: {policy}")
    compression = str(config.get("compression", "zstd"))
    if compression != "zstd":
        raise ValueError(f"Unsupported tiered_memory.compression: {compression}")
    ratio = float(config.get("tier2_budget_ratio", 0.25))
    budget_bytes = config.get("tier2_budget_bytes")
    if budget_bytes is None:
        memory_limit = _parse_memory_limit_bytes(os.getenv("RLM_MEM_LIMIT"))
        if memory_limit is None:
            memory_limit = 2 * 1024**3
        budget_bytes = int(memory_limit * ratio)
    spill_dir = config.get("spill_dir")
    if spill_dir is None:
        spill_dir = str(Path(tempfile.gettempdir()) / "rlm-py-wasm-tiered-memory")
    ttl_seconds = config.get("ttl_seconds", 3600)
    return {
        "policy": policy,
        "idle_grace_ms": int(config.get("idle_grace_ms", 2000)),
        "tier2_budget_ratio": ratio,
        "tier2_budget_bytes": int(budget_bytes),
        "spill_policy": str(config.get("spill_policy", "oldest_idle_first")),
        "restore_policy": str(config.get("restore_policy", "restore_on_next_turn")),
        "demote_only_at": str(config.get("demote_only_at", "quiescent_point")),
        "compression": compression,
        "compression_level": int(config.get("compression_level", 3)),
        "ttl_seconds": None if ttl_seconds is None else float(ttl_seconds),
        "spill_dir": str(spill_dir),
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
        callback_timeout_ms: int | None = None,
        max_fuel: int | None = None,
        deterministic_seed: int | None = None,
        restore_probe_on_idle: bool = False,
        tiered_memory: dict[str, Any] | None = None,
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
        self.callback_timeout_ms = (
            execution_timeout_ms if callback_timeout_ms is None else callback_timeout_ms
        )
        self.max_fuel = max_fuel
        self.deterministic_seed = deterministic_seed
        self.restore_probe_on_idle = restore_probe_on_idle
        self.tiered_memory_config = _normalize_tiered_memory_config(tiered_memory)
        self.tiered_memory_enabled = self.tiered_memory_config is not None
        if self.restore_probe_on_idle and self.tiered_memory_enabled:
            raise ValueError(
                "restore_probe_on_idle and tiered_memory are mutually exclusive"
            )
        self._lock = threading.Lock()
        self._timing_lock = threading.Lock()
        self._restore_probe_lock = threading.Lock()
        self._tiered_memory_lock = threading.Lock()
        self._pending_llm_calls: list[RLMChatCompletion] = []
        self._context_count = 0
        self._history_count = 0
        self._shadow_locals: dict[str, Any] = {}
        self._last_stats: dict[str, Any] = {}
        self._restore_probe_events: list[dict[str, Any]] = []
        self._restore_probe_skips: list[dict[str, Any]] = []
        self._restore_probe_pending: dict[str, Any] | None = None
        self._tiered_memory_events: list[dict[str, Any]] = []
        self._tiered_memory_skips: list[dict[str, Any]] = []
        self._tiered_memory_session_id = uuid.uuid4().hex
        self._tiered_memory_store = (
            _get_tiered_memory_store(self.tiered_memory_config)
            if self.tiered_memory_config is not None
            else None
        )
        self._tiered_memory_timer: threading.Timer | None = None
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
        self.session = self._new_session()
        if self._tiered_memory_store is not None:
            self._tiered_memory_store.mark_live(self._tiered_memory_session_id)
        self._pending_llm_calls = []
        self._shadow_locals = {}
        self._install_scaffold()
        self._prime_restore_probe_callback()

    def _new_session(self, *, linear_memory_initial_bytes: int | None = None):
        if self.eryx is None:
            raise RuntimeError("PyWasmEnv is missing the pyeryx module")

        callbacks = [
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
        ]
        if self._linear_memory_hooks_enabled():
            callbacks.append(
                {
                    "name": "__rlm_restore_probe_noop",
                    "fn": self._restore_probe_noop,
                    "description": "No-op callback used to initialize pyeryx callback state before restore",
                }
            )

        kwargs: dict[str, Any] = {
            "result_variable": "answer",
            "execution_timeout_ms": self.execution_timeout_ms,
            "callback_timeout_ms": self.callback_timeout_ms,
            "max_fuel": self.max_fuel,
            "callbacks": callbacks,
        }
        if self._linear_memory_hooks_enabled():
            kwargs["track_linear_memory"] = True
            if linear_memory_initial_bytes is not None:
                kwargs["linear_memory_initial_bytes"] = linear_memory_initial_bytes
        return self.eryx.Session(**kwargs)

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

        self._cancel_tiered_memory_timer()
        self._discard_tiered_memory_entry()
        if self.session is None:
            self.session = self._new_session()
            if self._tiered_memory_store is not None:
                self._tiered_memory_store.mark_live(self._tiered_memory_session_id)
        if self.session is not None:
            self.session.clear_state()
        self._context_count = 0
        self._history_count = 0
        self._shadow_locals = {}
        self._pending_llm_calls = []
        self._last_stats = {}
        self._restore_probe_events = []
        self._restore_probe_skips = []
        self._restore_probe_pending = None
        self._tiered_memory_events = []
        self._tiered_memory_skips = []
        self._timing = self._new_timing()
        self._last_execute_ended_at = None
        self._inter_turn_idle_started_at = None
        self._install_scaffold()
        self._prime_restore_probe_callback()

    def execute_code(self, code: str) -> REPLResult:
        if self.session is None and self.tiered_memory_enabled:
            self._restore_from_tiered_memory()
        if self.session is None:
            raise RuntimeError("PyWasmEnv is not initialized")

        start_time = time.perf_counter()
        park_before = self._timing_value("mid_execute_park_seconds")
        end_time = start_time
        with self._lock:
            self._pending_llm_calls = []
            try:
                self._install_scaffold()
                wrapped = self._wrap_user_code(code)
                result = self.session.execute(wrapped)
                end_time = time.perf_counter()
                self._record_execute_timing(start_time, end_time, park_before)
                stdout = result.stdout
                stderr = result.stderr
                answer = result.result if isinstance(result.result, dict) else None
                user_error = (
                    str(answer.get("__rlm_error", ""))
                    if answer is not None
                    else ""
                )
                final_answer = (
                    str(answer.get("content", ""))
                    if answer is not None and answer.get("ready") and not user_error
                    else None
                )
                self._last_stats = {
                    "duration_ms": result.duration_ms,
                    "callback_invocations": result.callback_invocations,
                    "peak_memory_bytes": result.peak_memory_bytes,
                    "fuel_consumed": result.fuel_consumed,
                    "result_error": result.result_error,
                    "idle_timing": self.runtime_timing(),
                    "restore_probe": self.restore_probe_stats(),
                    "tiered_memory": self.tiered_memory_stats(),
                }
                if result.result_error:
                    stderr = (stderr + "\n" if stderr else "") + result.result_error
                if user_error:
                    stderr = (stderr + "\n" if stderr else "") + user_error
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
                    "restore_probe": self.restore_probe_stats(),
                    "tiered_memory": self.tiered_memory_stats(),
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
        self._cancel_tiered_memory_timer()
        self._discard_tiered_memory_entry()
        if self._tiered_memory_store is not None:
            self._tiered_memory_store.unregister(self._tiered_memory_session_id)
        self.session = None
        self._shadow_locals.clear()
        self._pending_llm_calls.clear()
        self._restore_probe_pending = None

    def begin_inter_turn_idle(self) -> None:
        """Mark that the session is awaiting the next root-model response."""

        should_demote = False
        with self._timing_lock:
            if (
                self._last_execute_ended_at is not None
                and self._inter_turn_idle_started_at is None
            ):
                self._inter_turn_idle_started_at = self._last_execute_ended_at
                should_demote = True
        if should_demote and self.restore_probe_on_idle:
            if self._last_stats.get("result_error"):
                self._skip_restore_probe_demote("last_execute_result_error")
            else:
                self._demote_for_restore_probe()
        if should_demote and self.tiered_memory_enabled:
            if self._last_stats.get("result_error"):
                self._skip_tiered_memory_demote("last_execute_result_error")
            else:
                self._schedule_tiered_memory_demote()

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
        if self.restore_probe_on_idle:
            self._restore_from_restore_probe()
        if self.tiered_memory_enabled:
            self._cancel_tiered_memory_timer()
            self._restore_from_tiered_memory()

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

    def restore_probe_stats(self) -> dict[str, Any]:
        return {
            "enabled": self.restore_probe_on_idle,
            "event_count": len(self._restore_probe_events),
            "skip_count": len(self._restore_probe_skips),
            "has_pending_snapshot": self._restore_probe_pending is not None,
            "events": [dict(event) for event in self._restore_probe_events],
            "skips": [dict(skip) for skip in self._restore_probe_skips],
        }

    def tiered_memory_stats(self) -> dict[str, Any]:
        store_stats = (
            self._tiered_memory_store.stats()
            if self._tiered_memory_store is not None
            else {}
        )
        tier = (
            self._tiered_memory_store.tier_for(self._tiered_memory_session_id)
            if self._tiered_memory_store is not None
            else None
        )
        return {
            "enabled": self.tiered_memory_enabled,
            "policy": (
                dict(self.tiered_memory_config)
                if self.tiered_memory_config is not None
                else None
            ),
            "session_id": self._tiered_memory_session_id,
            "current_tier": tier,
            "timer_pending": self._tiered_memory_timer is not None,
            "event_count": len(self._tiered_memory_events),
            "skip_count": len(self._tiered_memory_skips),
            "events": [dict(event) for event in self._tiered_memory_events],
            "skips": [dict(skip) for skip in self._tiered_memory_skips],
            "store": store_stats,
        }

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup()
        return False

    def _execute_setup(self, code: str, *, session=None) -> None:
        target = self.session if session is None else session
        if target is None:
            raise RuntimeError("PyWasmEnv is not initialized")
        target.execute(code)

    def _install_scaffold(self, *, session=None) -> None:
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
""",
            session=session,
        )

    def _linear_memory_hooks_enabled(self) -> bool:
        return self.restore_probe_on_idle or self.tiered_memory_enabled

    def _prime_restore_probe_callback(self) -> None:
        if not self._linear_memory_hooks_enabled():
            return
        self._execute_setup("_restore_probe_noop = await __rlm_restore_probe_noop()")

    @staticmethod
    def _restore_probe_noop() -> str:
        return "ok"

    def _skip_restore_probe_demote(self, reason: str) -> None:
        result_error = str(self._last_stats.get("result_error") or "")
        self._restore_probe_skips.append(
            {
                "reason": reason,
                "result_error": result_error[:500],
            }
        )

    def _skip_tiered_memory_demote(self, reason: str) -> None:
        result_error = str(self._last_stats.get("result_error") or "")
        self._tiered_memory_skips.append(
            {
                "reason": reason,
                "result_error": result_error[:500],
            }
        )

    def _schedule_tiered_memory_demote(self) -> None:
        if self.tiered_memory_config is None:
            return
        grace_seconds = max(0, self.tiered_memory_config["idle_grace_ms"]) / 1000
        self._cancel_tiered_memory_timer()
        if grace_seconds == 0:
            self._demote_for_tiered_memory()
            return
        timer = threading.Timer(grace_seconds, self._demote_for_tiered_memory)
        timer.daemon = True
        self._tiered_memory_timer = timer
        timer.start()

    def _cancel_tiered_memory_timer(self) -> None:
        timer = self._tiered_memory_timer
        self._tiered_memory_timer = None
        if timer is not None:
            timer.cancel()

    def _discard_tiered_memory_entry(self) -> None:
        if self._tiered_memory_store is not None:
            self._tiered_memory_store.discard(self._tiered_memory_session_id)

    def _demote_for_tiered_memory(self) -> None:
        if self.tiered_memory_config is None or self._tiered_memory_store is None:
            return
        with self._tiered_memory_lock:
            self._tiered_memory_timer = None
            if self.session is None:
                return
            with self._timing_lock:
                if self._inter_turn_idle_started_at is None:
                    return
            snapshot_fn = getattr(self.session, "snapshot_linear_memory_regions", None)
            regions_fn = getattr(self.session, "linear_memory_regions", None)
            if not callable(snapshot_fn) or not callable(regions_fn):
                raise RuntimeError("tiered_memory requires pyeryx linear-memory hooks")

            import zstandard as zstd

            rss_before_demote = _current_process_rss_bytes()
            begin = time.perf_counter()
            before_regions = regions_fn()
            regions = snapshot_fn()
            snapshot_ms = (time.perf_counter() - begin) * 1000
            snapshot_bytes = sum(len(region["bytes"]) for region in regions)
            rss_after_snapshot = _current_process_rss_bytes()

            encoded = _encode_linear_memory_regions(regions)
            compression_begin = time.perf_counter()
            compressor = zstd.ZstdCompressor(
                level=int(self.tiered_memory_config["compression_level"])
            )
            compressed = compressor.compress(encoded)
            compression_ms = (time.perf_counter() - compression_begin) * 1000
            rss_after_compress = _current_process_rss_bytes()

            entry = _TieredMemoryEntry(
                session_id=self._tiered_memory_session_id,
                compressed=compressed,
                tier="tier2",
                created_at=time.monotonic(),
                last_access_at=time.monotonic(),
                disk_path=None,
                snapshot_bytes=snapshot_bytes,
                compressed_bytes=len(compressed),
                regions_before_demote=self._region_metadata(before_regions),
                snapshot_ms=snapshot_ms,
                compression_ms=compression_ms,
                rss_before_demote_bytes=rss_before_demote,
                rss_after_snapshot_bytes=rss_after_snapshot,
                rss_after_compress_bytes=rss_after_compress,
                rss_after_drop_live_bytes=None,
            )
            stored_tier = self._tiered_memory_store.put(entry)
            self.session = None
            del regions
            del encoded
            del compressed
            gc.collect()
            rss_after_drop_live = _current_process_rss_bytes()
            entry.rss_after_drop_live_bytes = rss_after_drop_live
            self._tiered_memory_events.append(
                {
                    "event": "demote",
                    "policy": self.tiered_memory_config["policy"],
                    "tier": stored_tier,
                    "trigger": "graceful_idle",
                    "snapshot_bytes": snapshot_bytes,
                    "compressed_bytes": entry.compressed_bytes,
                    "compression_ratio": (
                        entry.compressed_bytes / snapshot_bytes
                        if snapshot_bytes
                        else None
                    ),
                    "snapshot_ms": snapshot_ms,
                    "compression_ms": compression_ms,
                    "drop_live_ms": (time.perf_counter() - begin) * 1000,
                    "regions_before_demote": entry.regions_before_demote,
                    "rss_before_demote_bytes": rss_before_demote,
                    "rss_after_snapshot_bytes": rss_after_snapshot,
                    "rss_after_compress_bytes": rss_after_compress,
                    "rss_after_drop_live_bytes": rss_after_drop_live,
                    "rss_drop_after_drop_live_bytes": (
                        rss_before_demote - rss_after_drop_live
                        if (
                            rss_before_demote is not None
                            and rss_after_drop_live is not None
                        )
                        else None
                    ),
                    "store": self._tiered_memory_store.stats(),
                }
            )

    def _restore_from_tiered_memory(self) -> None:
        if self.tiered_memory_config is None or self._tiered_memory_store is None:
            return
        with self._tiered_memory_lock:
            if self.session is not None:
                return
            taken = self._tiered_memory_store.take(self._tiered_memory_session_id)
            if taken is None:
                return
            entry, compressed = taken
            import zstandard as zstd

            restore_begin = time.perf_counter()
            rss_before_restore = _current_process_rss_bytes()
            decompression_begin = time.perf_counter()
            decompressed = zstd.ZstdDecompressor().decompress(compressed)
            decompression_ms = (time.perf_counter() - decompression_begin) * 1000
            regions = _decode_linear_memory_regions(decompressed)
            rss_after_decompress = _current_process_rss_bytes()

            if len(regions) != 1:
                raise RuntimeError(
                    "tiered_memory currently supports exactly one linear-memory region"
                )
            snapshot_size = len(regions[0]["bytes"])
            calibration_session = self._new_session(
                linear_memory_initial_bytes=snapshot_size
            )
            calibration_regions = calibration_session.linear_memory_regions()
            if len(calibration_regions) != 1:
                raise RuntimeError(
                    "tiered_memory calibration expected one linear-memory region"
                )
            setup_delta = int(calibration_regions[0]["byte_size"]) - snapshot_size
            if setup_delta < 0:
                raise RuntimeError(
                    f"tiered_memory calibration produced negative setup delta {setup_delta}"
                )
            calibration_session = None
            gc.collect()
            rss_after_calibration_drop = _current_process_rss_bytes()

            initial_bytes = max(0, snapshot_size - setup_delta)
            restore_session = self._new_session(
                linear_memory_initial_bytes=initial_bytes
            )
            regions_before_restore = restore_session.linear_memory_regions()
            rss_after_restore_target_create = _current_process_rss_bytes()
            shape_matches = self._region_shape_metadata(regions_before_restore) == (
                self._region_shape_metadata(regions)
            )
            if not shape_matches:
                raise RuntimeError(
                    "tiered_memory failed to create a snapshot-shaped target: "
                    f"snapshot={self._region_shape_metadata(regions)} "
                    f"target={self._region_shape_metadata(regions_before_restore)}"
                )

            restore_stats = restore_session.restore_linear_memory_regions(regions)
            rss_after_restore_copy = _current_process_rss_bytes()
            self.session = restore_session
            self._repair_restore_probe_stdio(session=restore_session)
            rss_after_stdio_repair = _current_process_rss_bytes()
            self._tiered_memory_store.mark_live(self._tiered_memory_session_id)
            promote_latency_ms = (time.perf_counter() - restore_begin) * 1000
            self._tiered_memory_store.record_promote_latency_ms(promote_latency_ms)
            self._tiered_memory_events.append(
                {
                    "event": "promote",
                    "policy": self.tiered_memory_config["policy"],
                    "from_tier": entry.tier,
                    "snapshot_bytes": snapshot_size,
                    "compressed_bytes": entry.compressed_bytes,
                    "decompression_ms": decompression_ms,
                    "setup_delta_bytes": setup_delta,
                    "initial_bytes": initial_bytes,
                    "shape_matches_snapshot": shape_matches,
                    "restore_ms": promote_latency_ms,
                    "restore_stats": dict(restore_stats),
                    "regions_before_demote": entry.regions_before_demote,
                    "regions_before_restore": self._region_metadata(
                        regions_before_restore
                    ),
                    "rss_before_restore_bytes": rss_before_restore,
                    "rss_after_decompress_bytes": rss_after_decompress,
                    "rss_after_calibration_drop_bytes": rss_after_calibration_drop,
                    "rss_after_restore_target_create_bytes": (
                        rss_after_restore_target_create
                    ),
                    "rss_after_restore_copy_bytes": rss_after_restore_copy,
                    "rss_after_stdio_repair_bytes": rss_after_stdio_repair,
                    "store": self._tiered_memory_store.stats(),
                }
            )
            del compressed
            del decompressed
            del regions
            gc.collect()
            self._tiered_memory_events[-1]["rss_after_pending_release_bytes"] = (
                _current_process_rss_bytes()
            )

    def _demote_for_restore_probe(self) -> None:
        with self._restore_probe_lock:
            if self._restore_probe_pending is not None:
                return
            if self.session is None:
                return
            snapshot_fn = getattr(self.session, "snapshot_linear_memory_regions", None)
            regions_fn = getattr(self.session, "linear_memory_regions", None)
            if not callable(snapshot_fn) or not callable(regions_fn):
                raise RuntimeError(
                    "restore_probe_on_idle requires pyeryx linear-memory hooks"
                )

            rss_before_demote = _current_process_rss_bytes()
            begin = time.perf_counter()
            before_regions = regions_fn()
            regions = snapshot_fn()
            snapshot_ms = (time.perf_counter() - begin) * 1000
            snapshot_bytes = sum(len(region["bytes"]) for region in regions)
            rss_after_snapshot = _current_process_rss_bytes()
            self._restore_probe_pending = {
                "regions": regions,
                "snapshot_ms": snapshot_ms,
                "snapshot_bytes": snapshot_bytes,
                "regions_before_demote": self._region_metadata(before_regions),
                "rss_before_demote_bytes": rss_before_demote,
                "rss_after_snapshot_bytes": rss_after_snapshot,
            }
            self.session = None
            gc.collect()
            self._restore_probe_pending["drop_live_ms"] = (
                time.perf_counter() - begin
            ) * 1000
            rss_after_drop_live = _current_process_rss_bytes()
            self._restore_probe_pending["rss_after_drop_live_bytes"] = (
                rss_after_drop_live
            )
            self._restore_probe_pending["rss_drop_after_drop_live_bytes"] = (
                rss_before_demote - rss_after_drop_live
                if rss_before_demote is not None and rss_after_drop_live is not None
                else None
            )

    def _restore_from_restore_probe(self) -> None:
        with self._restore_probe_lock:
            pending = self._restore_probe_pending
            if pending is None:
                return
            regions = pending["regions"]
            if len(regions) != 1:
                raise RuntimeError(
                    "restore_probe_on_idle currently supports exactly one linear-memory region"
                )

            snapshot_size = len(regions[0]["bytes"])
            restore_begin = time.perf_counter()
            rss_before_restore = _current_process_rss_bytes()
            calibration_session = self._new_session(
                linear_memory_initial_bytes=snapshot_size
            )
            calibration_regions = calibration_session.linear_memory_regions()
            if len(calibration_regions) != 1:
                raise RuntimeError(
                    "restore_probe_on_idle calibration expected one linear-memory region"
                )
            setup_delta = int(calibration_regions[0]["byte_size"]) - snapshot_size
            if setup_delta < 0:
                raise RuntimeError(
                    f"restore_probe_on_idle calibration produced negative setup delta {setup_delta}"
                )
            calibration_session = None
            gc.collect()
            rss_after_calibration_drop = _current_process_rss_bytes()

            initial_bytes = max(0, snapshot_size - setup_delta)
            restore_session = self._new_session(
                linear_memory_initial_bytes=initial_bytes
            )
            regions_before_restore = restore_session.linear_memory_regions()
            rss_after_restore_target_create = _current_process_rss_bytes()
            shape_matches = self._region_shape_metadata(regions_before_restore) == (
                self._region_shape_metadata(regions)
            )
            if not shape_matches:
                raise RuntimeError(
                    "restore_probe_on_idle failed to create a snapshot-shaped target: "
                    f"snapshot={self._region_shape_metadata(regions)} "
                    f"target={self._region_shape_metadata(regions_before_restore)}"
                )

            restore_stats = restore_session.restore_linear_memory_regions(regions)
            rss_after_restore_copy = _current_process_rss_bytes()
            self.session = restore_session
            self._repair_restore_probe_stdio(session=restore_session)
            rss_after_stdio_repair = _current_process_rss_bytes()
            event = {
                "snapshot_bytes": snapshot_size,
                "snapshot_ms": pending["snapshot_ms"],
                "drop_live_ms": pending["drop_live_ms"],
                "setup_delta_bytes": setup_delta,
                "initial_bytes": initial_bytes,
                "shape_matches_snapshot": shape_matches,
                "restore_ms": (time.perf_counter() - restore_begin) * 1000,
                "restore_stats": dict(restore_stats),
                "regions_before_demote": pending["regions_before_demote"],
                "regions_before_restore": self._region_metadata(regions_before_restore),
                "rss_before_demote_bytes": pending.get("rss_before_demote_bytes"),
                "rss_after_snapshot_bytes": pending.get("rss_after_snapshot_bytes"),
                "rss_after_drop_live_bytes": pending.get("rss_after_drop_live_bytes"),
                "rss_drop_after_drop_live_bytes": pending.get(
                    "rss_drop_after_drop_live_bytes"
                ),
                "rss_before_restore_bytes": rss_before_restore,
                "rss_after_calibration_drop_bytes": rss_after_calibration_drop,
                "rss_after_restore_target_create_bytes": (
                    rss_after_restore_target_create
                ),
                "rss_after_restore_copy_bytes": rss_after_restore_copy,
                "rss_after_stdio_repair_bytes": rss_after_stdio_repair,
            }
            self._restore_probe_events.append(event)
            self._restore_probe_pending = None
            del pending
            del regions
            gc.collect()
            event["rss_after_pending_release_bytes"] = _current_process_rss_bytes()

    def _repair_restore_probe_stdio(self, *, session) -> None:
        self._execute_setup(
            """
import __main__ as _rlm_main

_rlm_stream_cls = type(_rlm_main._eryx_stdout)

def _rlm_stream_isatty(self):
    return False

def _rlm_stream_fileno(self):
    raise OSError("eryx streaming writer has no file descriptor")

_rlm_stream_cls.isatty = _rlm_stream_isatty
_rlm_stream_cls.fileno = _rlm_stream_fileno

def _rlm_safe_stream_value(stream, label):
    try:
        return stream.getvalue()
    except MemoryError:
        stream.reset()
        return f"[eryx {label} omitted: MemoryError while collecting output]"

def _rlm_get_output():
    _rlm_output = _rlm_safe_stream_value(_rlm_main._eryx_stdout, "stdout")
    _rlm_errors = _rlm_safe_stream_value(_rlm_main._eryx_stderr, "stderr")
    _rlm_main._sys.stdout = _rlm_main._eryx_old_stdout
    _rlm_main._sys.stderr = _rlm_main._eryx_old_stderr
    return _rlm_output, _rlm_errors

def _rlm_get_output_keep_capture():
    return (
        _rlm_safe_stream_value(_rlm_main._eryx_stdout, "stdout"),
        _rlm_safe_stream_value(_rlm_main._eryx_stderr, "stderr"),
    )

_rlm_main._eryx_get_output = _rlm_get_output
_rlm_main._eryx_get_output_keep_capture = _rlm_get_output_keep_capture
""",
            session=session,
        )

    @staticmethod
    def _region_metadata(regions: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return [
            {key: value for key, value in region.items() if key != "bytes"}
            for region in regions
        ]

    @staticmethod
    def _region_shape_metadata(regions: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return sorted(
            [
                {
                    "id": int(region["id"]),
                    "byte_capacity": int(region["byte_capacity"]),
                    "byte_size": int(region["byte_size"]),
                }
                for region in regions
            ],
            key=lambda region: region["id"],
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
                "    import sys as _rlm_sys\n"
                "    _rlm_numpy = _rlm_sys.modules.get('numpy')\n"
                "    if _rlm_numpy is not None:\n"
                f"        _rlm_numpy.random.seed({seed})\n"
                "except Exception:\n"
                "    pass\n"
            )
        body = textwrap.indent(rewritten, "    ") if rewritten.strip() else "    pass\n"
        return (
            prefix
            + "try:\n"
            + body
            + "\nexcept Exception:\n"
            + "    import traceback as _rlm_traceback\n"
            + "    answer = {\n"
            + "        'ready': False,\n"
            + "        'content': '',\n"
            + "        '__rlm_error': _rlm_traceback.format_exc(),\n"
            + "    }\n"
        )

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
