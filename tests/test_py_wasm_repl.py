import inspect

import pytest

eryx = pytest.importorskip("eryx")

from rlm.environments import get_environment
from rlm.environments.py_wasm_repl import (
    PyWasmEnv,
    py_wasm_tiered_memory_global_stats,
    rewrite_reserved_callback_calls,
)


def test_rewrite_reserved_callback_calls_wraps_sync_calls():
    rewritten = rewrite_reserved_callback_calls(
        'result = llm_query("hello", "model-a")\n'
        'already = await llm_query("done")\n'
    )

    assert "await llm_query(prompt='hello', model='model-a')" in rewritten
    assert "already = await llm_query(prompt='done')" in rewritten


def test_py_wasm_answer_and_context():
    env = PyWasmEnv(context_payload={"value": 42})
    try:
        result = env.execute_code(
            'answer["content"] = str(context["value"])\n'
            'answer["ready"] = True'
        )
        assert result.stderr == ""
        assert result.final_answer == "42"
        assert result.locals["answer"] == {"content": "42", "ready": True}
        assert result.locals["context"] == {"value": 42}
    finally:
        env.cleanup()


def test_py_wasm_llm_query_rewrite_without_handler():
    env = PyWasmEnv()
    try:
        result = env.execute_code('reply = llm_query("hello")\nprint(reply)')
        assert "Error: No LM handler configured" in result.stdout
        timing = result.locals["__pywasm_stats"]["idle_timing"]
        assert timing["mid_execute_park_count"] == 1
    finally:
        env.cleanup()


def test_py_wasm_user_exception_returns_stderr_without_poisoning_session():
    env = PyWasmEnv()
    try:
        result = env.execute_code("raise RuntimeError('boom')")
        assert "RuntimeError: boom" in result.stderr
        assert result.final_answer is None

        second = env.execute_code("print('ok after error')")
        assert second.stderr == ""
        assert second.stdout.strip() == "ok after error"
    finally:
        env.cleanup()


def test_py_wasm_restores_reserved_names_between_cells():
    env = PyWasmEnv(context_payload="original")
    try:
        env.execute_code('context = "hijacked"\nllm_query = lambda prompt: "bad"')
        result = env.execute_code("print(context)\nprint(await llm_query('x'))")
        assert "original" in result.stdout
        assert "Error: No LM handler configured" in result.stdout
    finally:
        env.cleanup()


def test_py_wasm_reset_for_request_clears_state_and_counts():
    env = PyWasmEnv(context_payload="first")
    try:
        env.execute_code("x = 42")
        env.begin_inter_turn_idle()
        env.end_inter_turn_idle()
        assert env.runtime_timing()["inter_turn_idle_count"] == 1
        assert env.get_context_count() == 1
        env.reset_for_request()
        assert env.get_context_count() == 0
        assert env.runtime_timing()["inter_turn_idle_count"] == 0
        result = env.execute_code("print('x' in globals())")
        assert result.stdout == "False"
    finally:
        env.cleanup()


def test_py_wasm_session_callback_timeout_defaults_to_execution_timeout():
    captured: list[dict] = []

    class FakeEryx:
        class Session:
            def __init__(self, **kwargs):
                captured.append(kwargs)

    env = PyWasmEnv(execution_timeout_ms=123_000)
    env.eryx = FakeEryx
    env._new_session()

    assert captured[-1]["execution_timeout_ms"] == 123_000
    assert captured[-1]["callback_timeout_ms"] == 123_000

    env = PyWasmEnv(execution_timeout_ms=123_000, callback_timeout_ms=45_000)
    env.eryx = FakeEryx
    env._new_session()

    assert captured[-1]["execution_timeout_ms"] == 123_000
    assert captured[-1]["callback_timeout_ms"] == 45_000


def test_py_wasm_restore_probe_round_trips_between_turns():
    if "linear_memory_initial_bytes" not in str(inspect.signature(eryx.Session)):
        pytest.skip("pyeryx fork restore-probe hooks are unavailable")

    env = PyWasmEnv(context_payload={"value": 7}, restore_probe_on_idle=True)
    try:
        first = env.execute_code(
            "payload = b'x' * (2 * 1024 * 1024)\n"
            "marker = context['value']\n"
            "print(len(payload))"
        )
        assert first.stderr == ""
        assert first.stdout.strip() == "2097152"

        env.begin_inter_turn_idle()
        assert env.restore_probe_stats()["has_pending_snapshot"] is True
        env.end_inter_turn_idle()

        stats = env.restore_probe_stats()
        assert stats["event_count"] == 1
        event = stats["events"][0]
        assert event["shape_matches_snapshot"] is True
        for key in (
            "rss_before_demote_bytes",
            "rss_after_snapshot_bytes",
            "rss_after_drop_live_bytes",
            "rss_before_restore_bytes",
            "rss_after_calibration_drop_bytes",
            "rss_after_restore_target_create_bytes",
            "rss_after_restore_copy_bytes",
            "rss_after_stdio_repair_bytes",
            "rss_after_pending_release_bytes",
        ):
            assert key in event
            assert event[key] is None or event[key] >= 0
        assert event["rss_drop_after_drop_live_bytes"] is None or isinstance(
            event["rss_drop_after_drop_live_bytes"], int
        )

        second = env.execute_code(
            "print(len(payload))\n"
            "print(marker)\n"
            "print(await llm_query('x'))"
        )
        assert second.stderr == ""
        assert second.stdout.strip().splitlines() == [
            "2097152",
            "7",
            "Error: No LM handler configured",
        ]
        assert second.locals["__pywasm_stats"]["restore_probe"]["event_count"] == 1
    finally:
        env.cleanup()


def test_py_wasm_restore_probe_deterministic_seed_does_not_import_numpy_after_restore():
    if "linear_memory_initial_bytes" not in str(inspect.signature(eryx.Session)):
        pytest.skip("pyeryx fork restore-probe hooks are unavailable")

    env = PyWasmEnv(
        context_payload={"value": 3},
        deterministic_seed=0,
        restore_probe_on_idle=True,
    )
    try:
        first = env.execute_code("payload = b'x' * 1024\nmarker = context['value']")
        assert first.stderr == ""

        env.begin_inter_turn_idle()
        env.end_inter_turn_idle()

        second = env.execute_code("print(marker)")
        assert second.stderr == ""
        assert second.stdout.strip() == "3"
        assert second.locals["__pywasm_stats"]["restore_probe"]["event_count"] == 1
    finally:
        env.cleanup()


def test_py_wasm_restore_probe_skips_demote_after_result_error():
    if "linear_memory_initial_bytes" not in str(inspect.signature(eryx.Session)):
        pytest.skip("pyeryx fork restore-probe hooks are unavailable")

    env = PyWasmEnv(restore_probe_on_idle=True)
    try:
        first = env.execute_code("marker = 1")
        assert first.stderr == ""
        env._last_stats["result_error"] = "ExecutionError: boom"

        env.begin_inter_turn_idle()
        stats = env.restore_probe_stats()
        assert stats["event_count"] == 0
        assert stats["skip_count"] == 1
        assert stats["skips"][0]["reason"] == "last_execute_result_error"
        assert stats["has_pending_snapshot"] is False

        env.end_inter_turn_idle()
        second = env.execute_code("print('still live')")
        assert second.stderr == ""
        assert second.stdout.strip() == "still live"
        assert second.locals["__pywasm_stats"]["restore_probe"]["skip_count"] == 1
    finally:
        env.cleanup()


def test_py_wasm_tiered_memory_spills_and_promotes_between_turns(tmp_path):
    if "linear_memory_initial_bytes" not in str(inspect.signature(eryx.Session)):
        pytest.skip("pyeryx fork tiered-memory hooks are unavailable")

    env = PyWasmEnv(
        context_payload={"value": 9},
        tiered_memory={
            "idle_grace_ms": 0,
            "tier2_budget_bytes": 1,
            "spill_dir": str(tmp_path),
            "ttl_seconds": 3600,
        },
    )
    try:
        first = env.execute_code(
            "payload = b'x' * (1024 * 1024)\n"
            "marker = context['value']\n"
            "print(len(payload))"
        )
        assert first.stderr == ""

        env.begin_inter_turn_idle()
        stats = env.tiered_memory_stats()
        assert stats["event_count"] == 1
        assert stats["events"][0]["event"] == "demote"
        assert stats["events"][0]["tier"] == "tier3"
        assert stats["current_tier"] == "tier3"
        assert stats["store"]["tier3_entries"] >= 1

        env.end_inter_turn_idle()
        stats = env.tiered_memory_stats()
        assert stats["event_count"] == 2
        assert stats["events"][1]["event"] == "promote"
        assert stats["events"][1]["from_tier"] == "tier3"
        assert stats["events"][1]["shape_matches_snapshot"] is True
        assert stats["current_tier"] == "tier1"

        second = env.execute_code("print(len(payload))\nprint(marker)")
        assert second.stderr == ""
        assert second.stdout.strip().splitlines() == ["1048576", "9"]
        task_stats = second.locals["__pywasm_stats"]["tiered_memory"]
        assert task_stats["event_count"] == 2

        global_stats = py_wasm_tiered_memory_global_stats()
        assert global_stats["demote_count"] >= 1
        assert global_stats["promote_count"] >= 1
        assert global_stats["spill_count"] >= 1
    finally:
        env.cleanup()


def test_get_environment_supports_py_wasm():
    env = get_environment("py_wasm", {})
    try:
        assert isinstance(env, PyWasmEnv)
    finally:
        env.cleanup()
