"""Request-scoped reset for warm session reuse (RLM Lab, PRD-007 slice 1).

Additive helper that lets a long-lived (``persistent=True``) :class:`LocalREPL`
be reused across benchmark tasks ("warm_reuse") while each task still sees a
clean request scope. Kept in its own module so ``local_repl.py`` is untouched
and the fork stays trivial to rebase on upstream.

What it resets (request scope):
  * the user namespace (``globals``/``locals``, rebuilt via ``setup()``),
  * the scratch ``temp_dir``,
  * the versioned context/history counters.

What it deliberately keeps warm: the live process and ``sys.modules`` —
re-importing is the cold-start cost warm reuse exists to avoid. Module-level
pollution this cannot undo (monkeypatching, module-level caches/config) is
handled one tier up by recycling the worker subprocess (PRD-007 Design
Decisions: warm_reuse borrows restart's fresh-process move on detected
pollution).
"""

import shutil
import tempfile
import uuid
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from rlm.environments.local_repl import LocalREPL


def reset_for_request(env: "LocalREPL") -> None:
    """Drop a persistent ``LocalREPL``'s request-scoped state, keeping it warm.

    Safe to call between completions on a reused environment: ``setup()``
    rebuilds fresh ``globals``/``locals`` dicts (no dangling references), and a
    new ``temp_dir`` isolates scratch files from the previous request.
    ``sys.modules`` is intentionally left warm. Mirrors the relevant parts of
    ``LocalREPL.__init__`` so a reused env matches a freshly created one.
    """
    # Fresh scratch dir — files written by the previous task must not leak.
    shutil.rmtree(env.temp_dir, ignore_errors=True)
    env.temp_dir = tempfile.mkdtemp(prefix=f"repl_env_{uuid.uuid4()}_")

    # Reset versioned context/history counters so the next add_context() starts
    # at context_0 (matches a fresh environment).
    env._context_count = 0
    env._history_count = 0

    # Rebuild the namespace + scaffold: clears user variables, any custom-tool
    # overwrites, the captured answer, and pending LLM calls.
    env.setup()

    # Re-init compaction history after setup(), matching __init__ ordering.
    if env.compaction:
        env._compaction_history = []
        env.locals["history"] = env._compaction_history

    # Reset per-request idle timing (RLM Lab, BR002-WO002) so a reused warm env
    # reports each task's idle structure like a fresh environment.
    reset_timing = getattr(env, "reset_idle_timing", None)
    if callable(reset_timing):
        reset_timing()
