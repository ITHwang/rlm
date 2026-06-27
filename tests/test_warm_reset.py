"""Tests for warm-reuse request reset (RLM Lab, PRD-007 slice 1)."""

import os
import sys

from rlm.environments.local_repl import LocalREPL
from rlm.environments.warm_reset import reset_for_request


def test_reset_clears_user_vars_and_restores_scaffold():
    repl = LocalREPL()
    repl.execute_code("x = 41\nanswer['content'] = 'leftover'")
    assert "x" in repl.locals

    reset_for_request(repl)

    # User variables are gone; the answer dict + reserved tools are fresh.
    assert "x" not in repl.locals
    assert "llm_query" in repl.globals
    assert repl.locals["answer"]["content"] == ""
    assert repl.locals["answer"]["ready"] is False
    repl.cleanup()


def test_reset_gives_fresh_temp_dir_without_leaked_files():
    repl = LocalREPL()
    old_dir = repl.temp_dir
    # execute_code runs with cwd = temp_dir, so this writes into the scratch dir.
    repl.execute_code("from pathlib import Path\nPath('leak.txt').write_text('x')")
    assert os.path.exists(os.path.join(old_dir, "leak.txt"))

    reset_for_request(repl)

    assert repl.temp_dir != old_dir
    assert os.path.isdir(repl.temp_dir)
    assert not os.path.exists(old_dir)  # old scratch dir removed
    assert not os.path.exists(os.path.join(repl.temp_dir, "leak.txt"))
    repl.cleanup()


def test_reset_zeros_context_and_history_counters():
    repl = LocalREPL()
    repl.add_context("first", 0)
    repl.add_context("second", 1)
    assert repl.get_context_count() == 2

    reset_for_request(repl)

    assert repl.get_context_count() == 0
    assert repl.get_history_count() == 0
    assert "context_0" not in repl.locals
    repl.cleanup()


def test_reset_keeps_sys_modules_warm():
    repl = LocalREPL()
    repl.execute_code("import math")
    assert "math" in sys.modules

    reset_for_request(repl)

    # sys.modules is process-global and deliberately NOT cleared by reset.
    assert "math" in sys.modules
    repl.cleanup()
