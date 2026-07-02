import pytest

pytest.importorskip("eryx")

from rlm.environments import get_environment
from rlm.environments.py_wasm_repl import PyWasmEnv, rewrite_reserved_callback_calls


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


def test_get_environment_supports_py_wasm():
    env = get_environment("py_wasm", {})
    try:
        assert isinstance(env, PyWasmEnv)
    finally:
        env.cleanup()
