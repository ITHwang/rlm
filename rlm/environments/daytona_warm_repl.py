"""Daytona warm-reuse REPL using Daytona's standard sandbox image.

This additive environment is for RLM Lab PRD-008. It keeps the upstream-style
broker/LMHandler bridge from :mod:`daytona_repl`, but adds lifecycle-aware reset
for either SDK code-interpreter contexts or the process executor. The default path
intentionally uses Daytona's default snapshot path so managed Daytona's standard
sandbox image/snapshot is used; explicit image params are reserved for
custom-image fallbacks.
"""

from __future__ import annotations

import base64
import contextlib
import gzip
import json
import os
import shlex
import textwrap
import threading
import time
from pathlib import Path
from typing import Any

import requests
from daytona import (
    CreateSandboxFromSnapshotParams,
    Daytona,
    DaytonaConfig,
    SessionExecuteRequest,
)
from requests.adapters import HTTPAdapter

from rlm.core.types import REPLResult
from rlm.environments.base_env import extract_tool_value
from rlm.environments.daytona_repl import _BROKER_SCRIPT, DaytonaREPL

_PROCESS_STATE_FILE = "/tmp/rlm_state.dill"
_PROCESS_DEFS_FILE = "/tmp/rlm_defs.json"
_CONTEXT_HOST_PATH_KEY = "__rlm_context_host_path__"
_CONTEXT_REMOTE_PATH_KEY = "__rlm_context_file__"
_UPLOAD_PARALLELISM = max(1, int(os.getenv("RLM_DAYTONA_UPLOAD_PARALLELISM", "1")))
_HOST_UPLOAD_SEMAPHORE = threading.BoundedSemaphore(_UPLOAD_PARALLELISM)
_BROKER_PENDING_ROUTE = textwrap.dedent(
    """
    @app.route("/pending")
    def get_pending():
        \"\"\"Called by DaytonaWarmREPL to long-poll pending requests.\"\"\"
        import time

        wait_seconds = min(float(request.args.get("wait", "20")), 30.0)
        deadline = time.monotonic() + wait_seconds
        while True:
            with lock:
                pending = [
                    {"id": rid, "request": entry["request"]}
                    for rid, entry in pending_requests.items()
                    if entry["response"] is None
                ]
            if pending or time.monotonic() >= deadline:
                return jsonify({"pending": pending})
            time.sleep(0.2)
    """
)
_BROKER_UPLOAD_ROUTE = textwrap.dedent(
    """
    @app.route("/upload_payload/<name>", methods=["PUT"])
    def upload_payload(name):
        from pathlib import Path

        if "/" in name or name.startswith("."):
            return jsonify({"error": "invalid payload name"}), 400
        data = request.get_data(cache=False)
        path = Path("/tmp") / name
        path.write_bytes(data)
        return jsonify({"path": str(path), "bytes": len(data)})

    @app.route("/upload_payload_chunk/<name>", methods=["POST"])
    def upload_payload_chunk(name):
        from pathlib import Path

        if "/" in name or name.startswith("."):
            return jsonify({"error": "invalid payload name"}), 400
        offset = int(request.headers.get("x-rlm-upload-offset", "0"))
        data = request.get_data(cache=False)
        path = Path("/tmp") / name
        if offset > 0 and (not path.exists() or path.stat().st_size < offset):
            return jsonify({"error": "invalid upload offset"}), 409
        mode = "r+b" if path.exists() else "wb"
        with path.open(mode) as handle:
            handle.seek(offset)
            handle.write(data)
            handle.truncate(offset + len(data))
        return jsonify({"path": str(path), "bytes": path.stat().st_size})
    """
)


def _trace(message: str) -> None:
    if os.getenv("RLM_DAYTONA_TRACE"):
        print(f"[daytona] {message}", flush=True)


def _build_context_exec_code(
    code: str,
    *,
    broker_port: int = 8080,
    depth: int = 1,
    request_timeout: int = 300,
    custom_tools: dict[str, Any] | None = None,
) -> str:
    """Build code that runs inside one Daytona code-interpreter context."""
    code_b64 = base64.b64encode(code.encode()).decode()
    custom_tools_code = ""
    if custom_tools:
        tool_lines: list[str] = []
        for name, entry in custom_tools.items():
            value = extract_tool_value(entry)
            if isinstance(value, str) and (
                value.strip().startswith(("def ", "class ", "lambda"))
                or "\n" in value
            ):
                tool_lines.append(f"# Custom tool: {name}")
                tool_lines.append(value)
                tool_lines.append(f"globals()[{name!r}] = {name}")
            else:
                try:
                    encoded = json.dumps(value)
                    tool_lines.append(f"globals()[{name!r}] = json.loads({encoded!r})")
                except (TypeError, ValueError):
                    tool_lines.append(f"# Warning: could not serialize tool {name!r}")
        custom_tools_code = "\n".join(tool_lines)

    return textwrap.dedent(
        f"""
        import base64
        import io
        import json
        import sys
        import traceback

        import requests

        BROKER_URL = "http://127.0.0.1:{broker_port}"

        def llm_query(prompt, model=None):
            try:
                response = requests.post(
                    f"{{BROKER_URL}}/enqueue",
                    json={{"type": "single", "prompt": prompt, "model": model, "depth": {depth}}},
                    timeout={request_timeout},
                )
                data = response.json()
                if data.get("error"):
                    return f"Error: {{data['error']}}"
                return data.get("response", "Error: No response")
            except Exception as exc:
                return f"Error: LM query failed - {{exc}}"

        def llm_query_batched(prompts, model=None):
            try:
                response = requests.post(
                    f"{{BROKER_URL}}/enqueue",
                    json={{"type": "batched", "prompts": prompts, "model": model, "depth": {depth}}},
                    timeout={request_timeout},
                )
                data = response.json()
                if data.get("error"):
                    return [f"Error: {{data['error']}}"] * len(prompts)
                return data.get("responses", ["Error: No response"] * len(prompts))
            except Exception as exc:
                return [f"Error: LM query failed - {{exc}}"] * len(prompts)

        def SHOW_VARS():
            available = {{
                k: type(v).__name__
                for k, v in globals().items()
                if not k.startswith("_")
                and k not in {{
                    "base64",
                    "io",
                    "json",
                    "requests",
                    "sys",
                    "traceback",
                    "llm_query",
                    "llm_query_batched",
                    "SHOW_VARS",
                    "answer",
                }}
            }}
            if not available:
                return "No variables created yet. Use ```repl``` blocks to create variables."
            return f"Available variables: {{available}}"

        if "answer" not in globals() or not isinstance(globals().get("answer"), dict):
            answer = {{"content": "", "ready": False}}

        {custom_tools_code}

        _rlm_code = base64.b64decode("{code_b64}").decode()
        _stdout_buf = io.StringIO()
        _stderr_buf = io.StringIO()
        _old_stdout, _old_stderr = sys.stdout, sys.stderr
        try:
            sys.stdout = _stdout_buf
            sys.stderr = _stderr_buf
            exec(_rlm_code, globals(), globals())
        except Exception:
            traceback.print_exc(file=_stderr_buf)
        finally:
            sys.stdout = _old_stdout
            sys.stderr = _old_stderr

        if "context_0" in globals():
            context = context_0
        if "history_0" in globals():
            history = history_0

        def _serialize(value):
            try:
                rendered = repr(value)
            except Exception:
                return f"<{{type(value).__name__}}>"
            if len(rendered) > 2000:
                return rendered[:2000] + "...<truncated>"
            return rendered

        _locals = {{
            k: _serialize(v)
            for k, v in globals().items()
            if not k.startswith("_")
            and k != "context"
            and not k.startswith("context_")
            and k != "history"
            and not k.startswith("history_")
            and k not in {{
                "base64",
                "io",
                "json",
                "requests",
                "sys",
                "traceback",
            }}
        }}
        _ans = answer if isinstance(globals().get("answer"), dict) else None
        _final = str(_ans.get("content", "")) if (_ans is not None and _ans.get("ready")) else None
        print("__RLM_RESULT__" + json.dumps({{
            "stdout": _stdout_buf.getvalue(),
            "stderr": _stderr_buf.getvalue(),
            "locals": _locals,
            "final_answer": _final,
        }}))
        """
    )


def _build_process_exec_code(
    code: str,
    *,
    broker_port: int = 8080,
    depth: int = 1,
    request_timeout: int = 300,
    custom_tools: dict[str, Any] | None = None,
    payload_paths: dict[str, dict[int, str | dict[str, str]]] | None = None,
) -> str:
    """Build process.exec code with a file-backed persistent namespace."""
    code_b64 = base64.b64encode(code.encode()).decode()
    payload_paths_json = json.dumps(payload_paths or {})
    custom_tools_code = ""
    if custom_tools:
        tool_lines: list[str] = []
        for name, entry in custom_tools.items():
            value = extract_tool_value(entry)
            if isinstance(value, str) and (
                value.strip().startswith(("def ", "class ", "lambda"))
                or "\n" in value
            ):
                tool_lines.append(f"# Custom tool: {name}")
                tool_lines.append(value)
                tool_lines.append(f"_globals[{name!r}] = {name}")
            else:
                try:
                    encoded = json.dumps(value)
                    tool_lines.append(f"_locals[{name!r}] = json.loads({encoded!r})")
                except (TypeError, ValueError):
                    tool_lines.append(f"# Warning: could not serialize tool {name!r}")
        custom_tools_code = "\n".join(tool_lines)

    return textwrap.dedent(
        f"""
        import base64
        import ast
        import importlib
        import io
        import json
        import os
        import sys
        import traceback
        import types
        from pathlib import Path

        import requests

        try:
            import dill
        except ImportError:
            import pickle as dill

        BROKER_URL = "http://127.0.0.1:{broker_port}"
        STATE_FILE = {_PROCESS_STATE_FILE!r}
        DEFS_FILE = {_PROCESS_DEFS_FILE!r}
        PAYLOAD_PATHS = json.loads({payload_paths_json!r})

        def llm_query(prompt, model=None):
            try:
                response = requests.post(
                    f"{{BROKER_URL}}/enqueue",
                    json={{"type": "single", "prompt": prompt, "model": model, "depth": {depth}}},
                    timeout={request_timeout},
                )
                data = response.json()
                if data.get("error"):
                    return f"Error: {{data['error']}}"
                return data.get("response", "Error: No response")
            except Exception as exc:
                return f"Error: LM query failed - {{exc}}"

        def llm_query_batched(prompts, model=None):
            try:
                response = requests.post(
                    f"{{BROKER_URL}}/enqueue",
                    json={{"type": "batched", "prompts": prompts, "model": model, "depth": {depth}}},
                    timeout={request_timeout},
                )
                data = response.json()
                if data.get("error"):
                    return [f"Error: {{data['error']}}"] * len(prompts)
                return data.get("responses", ["Error: No response"] * len(prompts))
            except Exception as exc:
                return [f"Error: LM query failed - {{exc}}"] * len(prompts)

        def load_state():
            if os.path.exists(STATE_FILE):
                try:
                    with open(STATE_FILE, "rb") as handle:
                        return dill.load(handle)
                except Exception:
                    pass
            return {{}}

        def load_definitions():
            if os.path.exists(DEFS_FILE):
                try:
                    with open(DEFS_FILE, "r", encoding="utf-8") as handle:
                        data = json.load(handle)
                    if isinstance(data, list):
                        return [item for item in data if isinstance(item, str)]
                except Exception:
                    pass
            return []

        def save_definitions(code, existing):
            try:
                tree = ast.parse(code)
            except SyntaxError:
                return
            definitions = list(existing)
            for node in tree.body:
                if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                    continue
                segment = ast.get_source_segment(code, node)
                if segment and segment not in definitions:
                    definitions.append(segment)
            if definitions != existing:
                with open(DEFS_FILE, "w", encoding="utf-8") as handle:
                    json.dump(definitions, handle)

        def save_state(state):
            clean_state = {{}}
            module_names = {{}}
            for key, value in state.items():
                if key == "__rlm_imported_modules__":
                    continue
                if isinstance(value, types.ModuleType):
                    module_names[key] = value.__name__
                    continue
                if key.startswith("_"):
                    continue
                if key == "context" or key.startswith("context_"):
                    continue
                if key == "history" or key.startswith("history_"):
                    continue
                try:
                    dill.dumps(value)
                    clean_state[key] = value
                except Exception:
                    pass
            if module_names:
                clean_state["__rlm_imported_modules__"] = module_names
            with open(STATE_FILE, "wb") as handle:
                dill.dump(clean_state, handle)

        def _serialize(value):
            try:
                rendered = repr(value)
            except Exception:
                return f"<{{type(value).__name__}}>"
            if len(rendered) > 2000:
                return rendered[:2000] + "...<truncated>"
            return rendered

        _locals = load_state()
        for _module_key, _module_name in _locals.pop(
            "__rlm_imported_modules__", {{}}
        ).items():
            try:
                _locals[_module_key] = importlib.import_module(_module_name)
            except Exception:
                pass
        for _prefix, _paths in PAYLOAD_PATHS.items():
            for _index_text, _payload_spec in _paths.items():
                _payload_key = f"{{_prefix}}_{{int(_index_text)}}"
                if isinstance(_payload_spec, dict):
                    _path = _payload_spec.get("path")
                    _format = _payload_spec.get("format", "json")
                else:
                    _path = _payload_spec
                    _format = "json"
                if _format == "text":
                    _locals[_payload_key] = Path(_path).read_text()
                else:
                    _locals[_payload_key] = json.loads(Path(_path).read_text())
            if "0" in _paths:
                _locals[_prefix] = _locals[f"{{_prefix}}_0"]
        if "answer" not in _locals or not isinstance(_locals.get("answer"), dict):
            _locals["answer"] = {{"content": "", "ready": False}}

        def SHOW_VARS():
            available = {{
                k: type(v).__name__
                for k, v in _locals.items()
                if not k.startswith("_") and k != "answer"
            }}
            if not available:
                return "No variables created yet. Use ```repl``` blocks to create variables."
            return f"Available variables: {{available}}"

        _globals = {{
            "__builtins__": __builtins__,
            "__name__": "__main__",
            "ast": ast,
            "base64": base64,
            "importlib": importlib,
            "io": io,
            "json": json,
            "os": os,
            "Path": Path,
            "requests": requests,
            "sys": sys,
            "traceback": traceback,
            "types": types,
            "llm_query": llm_query,
            "llm_query_batched": llm_query_batched,
            "SHOW_VARS": SHOW_VARS,
        }}

        {custom_tools_code}

        _rlm_code = base64.b64decode("{code_b64}").decode()
        _stdout_buf = io.StringIO()
        _stderr_buf = io.StringIO()
        _old_stdout, _old_stderr = sys.stdout, sys.stderr
        try:
            sys.stdout = _stdout_buf
            sys.stderr = _stderr_buf
            combined = {{**_globals, **_locals}}
            _definitions = load_definitions()
            for _definition in _definitions:
                exec(_definition, combined, combined)
            exec(_rlm_code, combined, combined)
            save_definitions(_rlm_code, _definitions)
            for key, value in combined.items():
                if key not in _globals and not key.startswith("_"):
                    _locals[key] = value
        except Exception:
            traceback.print_exc(file=_stderr_buf)
        finally:
            sys.stdout = _old_stdout
            sys.stderr = _old_stderr

        if "context_0" in _locals:
            _locals["context"] = _locals["context_0"]
        if "history_0" in _locals:
            _locals["history"] = _locals["history_0"]

        save_state(_locals)

        _out_locals = {{
            k: _serialize(v)
            for k, v in _locals.items()
            if not k.startswith("_")
            and k != "context"
            and not k.startswith("context_")
            and k != "history"
            and not k.startswith("history_")
        }}
        _ans = _locals.get("answer") if isinstance(_locals.get("answer"), dict) else None
        _final = str(_ans.get("content", "")) if (_ans is not None and _ans.get("ready")) else None
        print("__RLM_RESULT__" + json.dumps({{
            "stdout": _stdout_buf.getvalue(),
            "stderr": _stderr_buf.getvalue(),
            "locals": _out_locals,
            "final_answer": _final,
        }}))
        """
    )


class DaytonaWarmREPL(DaytonaREPL):
    """Daytona environment with request-scoped code-interpreter reset."""

    def __init__(
        self,
        *args,
        use_default_sandbox: bool = True,
        execution_mode: str = "code_interpreter",
        broker_timeout_seconds: int = 300,
        sandbox_labels: dict[str, str] | None = None,
        **kwargs,
    ):
        self.use_default_sandbox = use_default_sandbox
        if execution_mode not in {"code_interpreter", "process"}:
            raise ValueError(
                "execution_mode must be 'code_interpreter' or 'process', "
                f"got {execution_mode!r}"
            )
        self.execution_mode = execution_mode
        self.broker_timeout_seconds = broker_timeout_seconds
        self._code_context = None
        self._context_count = 0
        self._history_count = 0
        self._baseline_memory_cgroup = {}
        self._process_payload_paths: dict[str, dict[int, str | dict[str, str]]] = {
            "context": {},
            "history": {},
        }
        self._staged_remote_files: set[str] = set()
        self._last_repl_result: dict[str, Any] | None = None
        self._poll_interval_seconds = float(
            os.getenv("RLM_DAYTONA_POLL_INTERVAL_SECONDS", "0.5")
        )
        self._pending_wait_seconds = float(
            os.getenv("RLM_DAYTONA_PENDING_WAIT_SECONDS", "20")
        )
        self._http = requests.Session()
        adapter = HTTPAdapter(pool_connections=4, pool_maxsize=4, pool_block=True)
        self._http.mount("http://", adapter)
        self._http.mount("https://", adapter)
        self.sandbox_labels = sandbox_labels or {}
        super().__init__(*args, **kwargs)

    def setup(self):
        _trace(f"setup start mode={self.execution_mode}")
        config_kwargs = {"target": self.target}
        if self.api_key:
            config_kwargs["api_key"] = self.api_key
        config = DaytonaConfig(**config_kwargs)
        self.daytona = Daytona(config)

        if self.use_default_sandbox:
            if (self.cpu, self.memory, self.disk) != (1, 1, 3):
                raise ValueError(
                    "Daytona default sandbox mode only supports the standard "
                    "1 vCPU / 1 GiB / 3 GiB shape. Use an explicit image/snapshot "
                    "for custom resources."
                )
            params = CreateSandboxFromSnapshotParams(
                name=self.name,
                labels=self.sandbox_labels,
                auto_stop_interval=self.auto_stop_interval,
            )
            _trace(f"create default sandbox start name={self.name}")
            self.sandbox = self.daytona.create(params, timeout=self.timeout)
            _trace("create default sandbox done")
        else:
            _trace("explicit image setup start")
            super().setup()
            _trace("explicit image setup done")
            self._ensure_process_dependencies()
            self._reset_execution_state()
            self._capture_baseline_memory()
            return

        with contextlib.suppress(Exception):
            self.sandbox.set_autostop_interval(self.auto_stop_interval)
        self._start_broker()
        self._ensure_process_dependencies()
        self._reset_execution_state()
        self._capture_baseline_memory()
        _trace("setup done")

    def _ensure_process_dependencies(self) -> None:
        if self.execution_mode != "process":
            return
        _trace("process dependency check start")
        response = self.sandbox.process.exec("python -c 'import dill'", timeout=30)
        if getattr(response, "exit_code", 1) == 0:
            _trace("process dependency check done dill=present")
            return
        _trace("process dependency install start package=dill")
        response = self.sandbox.process.exec(
            "python -m pip install --quiet 'dill>=0.3.7'",
            timeout=self.timeout,
        )
        if getattr(response, "exit_code", 1) != 0:
            raise RuntimeError(str(getattr(response, "result", "")))
        _trace("process dependency install done package=dill")

    def _start_broker(self) -> None:
        _trace("broker start")
        broker_script = _BROKER_SCRIPT.replace(
            "event.wait(timeout=300)",
            f"event.wait(timeout={int(self.broker_timeout_seconds)})",
        )
        original_pending_route = textwrap.dedent(
            '''
            @app.route("/pending")
            def get_pending():
                """Called by DaytonaREPL to get pending requests."""
                with lock:
                    pending = [
                        {"id": rid, "request": entry["request"]}
                        for rid, entry in pending_requests.items()
                        if entry["response"] is None
                    ]
                return jsonify({"pending": pending})
            '''
        )
        broker_script = broker_script.replace(
            original_pending_route,
            _BROKER_PENDING_ROUTE,
        )
        broker_script = broker_script.replace(
            '\nif __name__ == "__main__":',
            f"\n{_BROKER_UPLOAD_ROUTE}\nif __name__ == \"__main__\":",
        )
        self.sandbox.fs.upload_file(broker_script.encode("utf-8"), "broker_server.py")
        with contextlib.suppress(Exception):
            self.sandbox.process.delete_session(self.broker_session_id)
        self.sandbox.process.create_session(self.broker_session_id)
        self.sandbox.process.execute_session_command(
            self.broker_session_id,
            SessionExecuteRequest(command="python broker_server.py", var_async=True),
        )
        time.sleep(3)
        preview_info = self.sandbox.get_preview_link(self.BROKER_PORT)
        self.broker_url = preview_info.url
        self._preview_token = preview_info.token
        self._ensure_poller()
        _trace("broker ready")

    def _ensure_poller(self) -> None:
        if not (self.lm_handler_address and self.broker_url):
            return
        if self.poller_thread is not None and self.poller_thread.is_alive():
            return
        self.poller_stop.clear()
        import threading

        self.poller_thread = threading.Thread(target=self._poll_broker, daemon=True)
        self.poller_thread.start()

    def _poll_broker(self):
        """Poll the sandbox broker using a persistent host-side HTTP session."""
        while not self.poller_stop.is_set():
            try:
                response = self._request_with_retries(
                    "GET",
                    f"{self.broker_url}/pending",
                    headers=self._get_headers(),
                    params={"wait": self._pending_wait_seconds},
                    timeout=max(10, int(self._pending_wait_seconds) + 5),
                    attempts=1,
                )
                pending = response.json().get("pending", [])

                for item in pending:
                    request_id = item["id"]
                    req_data = item["request"]
                    try:
                        lm_response = self._handle_llm_request(req_data)
                    except Exception as exc:  # noqa: BLE001 - unblock broker waiters.
                        _trace(f"llm request handling failed: {exc}")
                        lm_response = {"error": f"LLM request failed: {exc}"}
                    self._request_with_retries(
                        "POST",
                        f"{self.broker_url}/respond",
                        headers=self._get_headers(),
                        json={"id": request_id, "response": lm_response},
                        timeout=10,
                        attempts=3,
                    )
            except requests.RequestException as exc:
                _trace(f"broker poll request failed: {exc}")
            except Exception as exc:
                _trace(f"broker poll failed: {exc}")

            time.sleep(self._poll_interval_seconds)

    def _stop_poller(self) -> None:
        if self.poller_thread is not None:
            self.poller_stop.set()
            self.poller_thread.join(timeout=2)
            self.poller_thread = None

    def _new_code_context(self) -> None:
        if self.execution_mode == "process":
            self._reset_process_state()
            return
        _trace("create code context start")
        if self._code_context is not None:
            with contextlib.suppress(Exception):
                self.sandbox.code_interpreter.delete_context(self._code_context)
        self._code_context = self.sandbox.code_interpreter.create_context()
        self._context_count = 0
        self._history_count = 0
        _trace("create code context done")

    def _reset_execution_state(self) -> None:
        if self.execution_mode == "process":
            self._reset_process_state()
        else:
            self._new_code_context()

    def _reset_process_state(self) -> None:
        if self.sandbox is not None:
            with contextlib.suppress(Exception):
                self.sandbox.process.exec(
                    f"sh -lc 'rm -f {_PROCESS_STATE_FILE} {_PROCESS_DEFS_FILE}'",
                    timeout=30,
                )
        self._context_count = 0
        self._history_count = 0
        self._process_payload_paths = {"context": {}, "history": {}}
        _trace("process state reset")

    def _capture_baseline_memory(self) -> None:
        _trace("baseline memory read start")
        with contextlib.suppress(Exception):
            self._baseline_memory_cgroup = self.read_memory_cgroup()
        _trace("baseline memory read done")

    def update_handler_address(self, address: tuple[str, int]) -> None:
        self.lm_handler_address = address
        self._ensure_poller()

    def add_context(
        self, context_payload: dict | list | str, context_index: int | None = None
    ) -> int:
        self._new_code_context()
        index = 0 if context_index is None else context_index
        self._load_payload("context", context_payload, index)
        self._context_count = max(self._context_count, index + 1)
        return index

    def get_context_count(self) -> int:
        return self._context_count

    def add_history(
        self, message_history: list[dict[str, Any]], history_index: int | None = None
    ) -> int:
        index = self._history_count if history_index is None else history_index
        self._load_payload("history", message_history, index)
        self._history_count = max(self._history_count, index + 1)
        return index

    def get_history_count(self) -> int:
        return self._history_count

    def get_last_repl_result(self) -> dict[str, Any] | None:
        return self._last_repl_result

    def load_context(self, context_payload: dict | list | str):
        index = self._context_count
        self._load_payload("context", context_payload, index)
        self._context_count = max(self._context_count, index + 1)

    def _load_payload(self, prefix: str, payload: dict | list | str, index: int) -> None:
        file_spec = self._payload_file_spec(payload)
        if file_spec is not None:
            remote_path, host_path = file_spec
            if host_path is not None:
                self._stage_host_file(host_path, remote_path)
            if self.execution_mode == "process":
                self._process_payload_paths.setdefault(prefix, {})[index] = {
                    "path": remote_path,
                    "format": "text",
                }
                _trace(f"load payload done prefix={prefix} via=file")
                return
            code = (
                "from pathlib import Path\n"
                f"{prefix}_{index} = Path({remote_path!r}).read_text()\n"
                f"{prefix} = {prefix}_0\n"
            )
            self.execute_code(code)
            _trace(f"load payload done prefix={prefix} via=file")
            return

        payload_json = json.dumps(payload)
        _trace(f"load payload start prefix={prefix} bytes={len(payload_json)}")
        if self.execution_mode == "process":
            remote_path = f"/tmp/rlm_{prefix}_{index}.json"
            self._upload_payload_json(payload_json, remote_path)
            self._process_payload_paths.setdefault(prefix, {})[index] = remote_path
            _trace(f"load payload done prefix={prefix}")
            return
        if len(payload_json) > 256 * 1024:
            remote_path = f"/tmp/rlm_{prefix}_{index}.json"
            self._upload_payload_json(payload_json, remote_path)
            code = (
                "import json\n"
                "from pathlib import Path\n"
                f"{prefix}_{index} = json.loads(Path({remote_path!r}).read_text())\n"
                f"{prefix} = {prefix}_0\n"
            )
        else:
            code = (
                "import json\n"
                f"{prefix}_{index} = json.loads({payload_json!r})\n"
                f"{prefix} = {prefix}_0\n"
            )
        self.execute_code(code)
        _trace(f"load payload done prefix={prefix}")

    def _payload_file_spec(
        self,
        payload: dict | list | str,
    ) -> tuple[str, str | None] | None:
        if not isinstance(payload, dict):
            return None
        remote_path = payload.get(_CONTEXT_REMOTE_PATH_KEY)
        if not isinstance(remote_path, str) or not remote_path.startswith("/tmp/"):
            return None
        host_path = payload.get(_CONTEXT_HOST_PATH_KEY)
        if host_path is not None and not isinstance(host_path, str):
            raise TypeError(f"{_CONTEXT_HOST_PATH_KEY} must be a string path")
        return remote_path, host_path

    def _stage_host_file(self, host_path: str, remote_path: str) -> None:
        if remote_path in self._staged_remote_files:
            return
        payload_bytes = Path(host_path).read_bytes()
        _trace(
            f"stage host file start path={remote_path} bytes={len(payload_bytes)}"
        )
        with _HOST_UPLOAD_SEMAPHORE:
            self._upload_payload_bytes(payload_bytes, remote_path)
        self._staged_remote_files.add(remote_path)
        _trace(f"stage host file done path={remote_path}")

    def _upload_payload_json(self, payload_json: str, remote_path: str) -> None:
        with _HOST_UPLOAD_SEMAPHORE:
            self._upload_payload_json_unlocked(payload_json, remote_path)

    def _upload_payload_json_unlocked(self, payload_json: str, remote_path: str) -> None:
        _trace(f"upload payload start path={remote_path}")
        self._upload_payload_bytes(payload_json.encode("utf-8"), remote_path)

    def _upload_payload_bytes(self, payload_bytes: bytes, remote_path: str) -> None:
        if len(payload_bytes) > 512 * 1024:
            try:
                self._upload_payload_via_fs_gzip(payload_bytes, remote_path)
                return
            except Exception as exc:  # noqa: BLE001 - fall back across SDK paths.
                _trace(
                    f"fs gzip upload failed path={remote_path}; "
                    f"fallback=broker error={exc}"
                )
        if self.broker_url:
            name = remote_path.rsplit("/", 1)[-1]
            try:
                if len(payload_bytes) <= 512 * 1024:
                    self._upload_payload_via_broker_put(payload_bytes, name)
                else:
                    self._upload_payload_via_broker_gzip_chunks(
                        payload_bytes,
                        name,
                        remote_path,
                    )
                _trace(f"upload payload done path={remote_path} via=broker")
                return
            except requests.RequestException as exc:
                _trace(
                    f"broker upload failed path={remote_path}; "
                    f"fallback=exec_chunks error={exc}"
                )
                self._upload_payload_via_exec_chunks(payload_bytes, remote_path)
                return
        self.sandbox.fs.upload_file(payload_bytes, remote_path)
        _trace(f"upload payload done path={remote_path} via=fs")

    def _upload_payload_via_broker_put(self, payload_bytes: bytes, name: str) -> None:
        upload_timeout = max(10, min(int(self.broker_timeout_seconds), 60))
        headers = {
            **self._get_headers(),
            "Content-Type": "application/octet-stream",
        }
        response = self._request_with_retries(
            "PUT",
            f"{self.broker_url}/upload_payload/{name}",
            headers=headers,
            data=payload_bytes,
            timeout=upload_timeout,
        )
        response.raise_for_status()

    def _upload_payload_via_broker_chunks(self, payload_bytes: bytes, name: str) -> None:
        chunk_size = 1024 * 1024
        upload_timeout = max(10, min(int(self.broker_timeout_seconds), 60))
        headers = {
            **self._get_headers(),
            "Content-Type": "application/octet-stream",
        }
        for offset in range(0, len(payload_bytes), chunk_size):
            chunk = payload_bytes[offset : offset + chunk_size]
            response = self._request_with_retries(
                "POST",
                f"{self.broker_url}/upload_payload_chunk/{name}",
                headers={
                    **headers,
                    "x-rlm-upload-offset": str(offset),
                },
                data=chunk,
                timeout=upload_timeout,
            )
            response.raise_for_status()
            if offset and offset % (5 * 1024 * 1024) == 0:
                _trace(
                    f"upload payload progress path=/tmp/{name} "
                    f"bytes={offset}"
                )

    def _upload_payload_via_broker_gzip_chunks(
        self,
        payload_bytes: bytes,
        name: str,
        remote_path: str,
    ) -> None:
        compressed = gzip.compress(payload_bytes, compresslevel=1)
        compressed_name = f"{name}.gz"
        compressed_path = f"{remote_path}.gz"
        self._upload_payload_via_broker_chunks(compressed, compressed_name)
        self._decompress_remote_gzip(compressed_path, remote_path)
        _trace(
            f"upload payload decompressed path={remote_path} "
            f"bytes={len(payload_bytes)} compressed_bytes={len(compressed)}"
        )

    def _upload_payload_via_fs_gzip(
        self,
        payload_bytes: bytes,
        remote_path: str,
    ) -> None:
        compressed = gzip.compress(payload_bytes, compresslevel=1)
        compressed_path = f"{remote_path}.gz"
        self.sandbox.fs.upload_file(compressed, compressed_path)
        self._decompress_remote_gzip(compressed_path, remote_path)
        _trace(
            f"upload payload done path={remote_path} via=fs_gzip "
            f"bytes={len(payload_bytes)} compressed_bytes={len(compressed)}"
        )

    def _decompress_remote_gzip(self, compressed_path: str, remote_path: str) -> None:
        decompress_code = (
            "from pathlib import Path\n"
            "import gzip\n"
            f"source = Path({compressed_path!r})\n"
            f"target = Path({remote_path!r})\n"
            "target.write_bytes(gzip.decompress(source.read_bytes()))\n"
            "source.unlink()\n"
        )
        response = self._process_exec_with_retries(
            f"python -c {shlex.quote(decompress_code)}",
            timeout=self.timeout,
        )
        if getattr(response, "exit_code", 1) != 0:
            raise RuntimeError(str(getattr(response, "result", "")))

    def _request_with_retries(self, method: str, url: str, **kwargs):
        attempts = int(kwargs.pop("attempts", 5))
        for attempt in range(1, attempts + 1):
            try:
                return self._http.request(method, url, **kwargs)
            except requests.RequestException:
                if attempt >= attempts:
                    raise
                _trace(f"http retry method={method} attempt={attempt + 1}/{attempts}")
                time.sleep(min(2 * attempt, 10))
        raise RuntimeError("unreachable http retry state")

    def _upload_payload_via_fs_chunks(
        self, payload_bytes: bytes, remote_path: str
    ) -> None:
        chunk_size = 512 * 1024
        part_paths: list[str] = []
        for part_index, offset in enumerate(range(0, len(payload_bytes), chunk_size)):
            part_path = f"{remote_path}.part{part_index:05d}"
            self.sandbox.fs.upload_file(
                payload_bytes[offset : offset + chunk_size],
                part_path,
            )
            part_paths.append(part_path)
            if offset and offset % (5 * 1024 * 1024) == 0:
                _trace(f"upload payload progress path={remote_path} bytes={offset}")
        concat_code = (
            "from pathlib import Path\n"
            f"target = Path({remote_path!r})\n"
            f"parts = {part_paths!r}\n"
            "with target.open('wb') as out:\n"
            "    for part in parts:\n"
            "        path = Path(part)\n"
            "        out.write(path.read_bytes())\n"
            "        path.unlink()\n"
        )
        response = self.sandbox.process.exec(
            f"python -c {shlex.quote(concat_code)}",
            timeout=self.timeout,
        )
        if getattr(response, "exit_code", 1) != 0:
            raise RuntimeError(str(getattr(response, "result", "")))
        _trace(
            f"upload payload done path={remote_path} via=fs_chunks "
            f"bytes={len(payload_bytes)} parts={len(part_paths)}"
        )

    def _upload_payload_via_exec_chunks(
        self, payload_bytes: bytes, remote_path: str
    ) -> None:
        chunk_size = 48 * 1024
        compressed = gzip.compress(payload_bytes, compresslevel=1)
        compressed_path = f"{remote_path}.gz"
        if len(compressed) <= chunk_size:
            encoded = base64.b64encode(compressed).decode()
            write_code = (
                "from pathlib import Path\n"
                "import base64\n"
                "import gzip\n"
                f"data = base64.b64decode({encoded!r})\n"
                f"Path({remote_path!r}).write_bytes(gzip.decompress(data))\n"
            )
            response = self._process_exec_with_retries(
                f"python -c {shlex.quote(write_code)}",
                timeout=120,
            )
            if getattr(response, "exit_code", 1) != 0:
                raise RuntimeError(str(getattr(response, "result", "")))
            _trace(
                f"upload payload done path={remote_path} via=exec_single "
                f"bytes={len(payload_bytes)} compressed_bytes={len(compressed)}"
            )
            return
        init_code = (
            f"from pathlib import Path; Path({compressed_path!r}).write_bytes(b'')"
        )
        response = self._process_exec_with_retries(
            f"python -c {shlex.quote(init_code)}",
            timeout=30,
        )
        if getattr(response, "exit_code", 1) != 0:
            raise RuntimeError(str(getattr(response, "result", "")))
        for offset in range(0, len(compressed), chunk_size):
            encoded = base64.b64encode(compressed[offset : offset + chunk_size])
            append_code = (
                "from pathlib import Path\n"
                "import base64\n"
                f"data = base64.b64decode({encoded.decode()!r})\n"
                f"with Path({compressed_path!r}).open('r+b') as handle:\n"
                f"    handle.seek({offset})\n"
                "    handle.write(data)\n"
                f"    handle.truncate({offset + min(chunk_size, len(compressed) - offset)})\n"
            )
            response = self._process_exec_with_retries(
                f"python -c {shlex.quote(append_code)}",
                timeout=60,
            )
            if getattr(response, "exit_code", 1) != 0:
                raise RuntimeError(str(getattr(response, "result", "")))
            if offset and offset % (5 * 1024 * 1024) == 0:
                _trace(f"upload payload progress path={remote_path} bytes={offset}")
        decompress_code = (
            "from pathlib import Path\n"
            "import gzip\n"
            f"source = Path({compressed_path!r})\n"
            f"target = Path({remote_path!r})\n"
            "target.write_bytes(gzip.decompress(source.read_bytes()))\n"
            "source.unlink()\n"
        )
        response = self._process_exec_with_retries(
            f"python -c {shlex.quote(decompress_code)}",
            timeout=self.timeout,
        )
        if getattr(response, "exit_code", 1) != 0:
            raise RuntimeError(str(getattr(response, "result", "")))
        _trace(
            f"upload payload done path={remote_path} via=exec_chunks "
            f"bytes={len(payload_bytes)} compressed_bytes={len(compressed)}"
        )

    def _process_exec_with_retries(self, command: str, *, timeout: int, attempts: int = 5):
        for attempt in range(1, attempts + 1):
            try:
                return self.sandbox.process.exec(command, timeout=timeout)
            except Exception:
                if attempt >= attempts:
                    raise
                _trace(f"process.exec retry attempt={attempt + 1}/{attempts}")
                time.sleep(min(2 * attempt, 10))
        raise RuntimeError("unreachable process.exec retry state")

    def execute_code(self, code: str) -> REPLResult:
        start_time = time.perf_counter()
        _trace(f"execute start mode={self.execution_mode} code_bytes={len(code)}")
        self._ensure_poller()
        with self._calls_lock:
            self.pending_llm_calls.clear()

        if self.execution_mode == "process":
            return self._execute_process_code(code, start_time)

        script = _build_context_exec_code(
            code,
            broker_port=self.BROKER_PORT,
            depth=self.depth,
            request_timeout=self.broker_timeout_seconds,
            custom_tools=self.custom_tools,
        )
        response = self.sandbox.code_interpreter.run_code(
            script,
            context=self._code_context,
            timeout=self.timeout,
        )
        _trace("code_interpreter.run_code returned")
        stdout = str(getattr(response, "stdout", "") or "")
        stderr = str(getattr(response, "stderr", "") or "")
        error = getattr(response, "error", None)
        if error:
            stderr = f"{stderr}\n{error}".strip()

        with self._calls_lock:
            pending_calls = self.pending_llm_calls.copy()
            self.pending_llm_calls.clear()

        execution_time = time.perf_counter() - start_time
        result_line = None
        for line in reversed(stdout.strip().splitlines()):
            if line.startswith("__RLM_RESULT__"):
                result_line = line.removeprefix("__RLM_RESULT__")
                break
        if result_line is None:
            return self._remember_repl_result(
                REPLResult(
                    stdout=stdout,
                    stderr=stderr or "Failed to parse execution result",
                    locals={},
                    execution_time=execution_time,
                    rlm_calls=pending_calls,
                )
            )
        try:
            result = json.loads(result_line)
        except json.JSONDecodeError:
            return self._remember_repl_result(
                REPLResult(
                    stdout=stdout,
                    stderr=stderr or "Failed to parse execution result",
                    locals={},
                    execution_time=execution_time,
                    rlm_calls=pending_calls,
                )
            )
        return self._remember_repl_result(
            REPLResult(
                stdout=result.get("stdout", ""),
                stderr=result.get("stderr", "") + (("\n" + stderr) if stderr else ""),
                locals=result.get("locals", {}),
                execution_time=execution_time,
                rlm_calls=pending_calls,
                final_answer=result.get("final_answer"),
            )
        )

    def _execute_process_code(self, code: str, start_time: float) -> REPLResult:
        _trace(f"process execute build start code_bytes={len(code)}")
        script = _build_process_exec_code(
            code,
            broker_port=self.BROKER_PORT,
            depth=self.depth,
            request_timeout=self.broker_timeout_seconds,
            custom_tools=self.custom_tools,
            payload_paths=self._process_payload_paths,
        )
        script_path = "/tmp/rlm_exec_script.py"
        _trace(f"process script upload start bytes={len(script)}")
        with _HOST_UPLOAD_SEMAPHORE:
            self._upload_payload_bytes(script.encode("utf-8"), script_path)
        _trace("process script upload done")
        _trace("process.exec start")
        response = self.sandbox.process.exec(
            f"python {script_path}",
            timeout=self.timeout,
        )
        _trace("process.exec returned")
        raw = str(getattr(response, "result", "") or "")
        stdout = raw if getattr(response, "exit_code", 1) == 0 else ""
        stderr = raw if getattr(response, "exit_code", 1) != 0 else ""

        with self._calls_lock:
            pending_calls = self.pending_llm_calls.copy()
            self.pending_llm_calls.clear()

        execution_time = time.perf_counter() - start_time
        result_line = None
        for line in reversed(stdout.strip().splitlines()):
            if line.startswith("__RLM_RESULT__"):
                result_line = line.removeprefix("__RLM_RESULT__")
                break
        if result_line is None:
            return self._remember_repl_result(
                REPLResult(
                    stdout=stdout,
                    stderr=stderr or "Failed to parse execution result",
                    locals={},
                    execution_time=execution_time,
                    rlm_calls=pending_calls,
                )
            )
        try:
            result = json.loads(result_line)
        except json.JSONDecodeError:
            return self._remember_repl_result(
                REPLResult(
                    stdout=stdout,
                    stderr=stderr or "Failed to parse execution result",
                    locals={},
                    execution_time=execution_time,
                    rlm_calls=pending_calls,
                )
            )
        return self._remember_repl_result(
            REPLResult(
                stdout=result.get("stdout", ""),
                stderr=result.get("stderr", "") + (("\n" + stderr) if stderr else ""),
                locals=result.get("locals", {}),
                execution_time=execution_time,
                rlm_calls=pending_calls,
                final_answer=result.get("final_answer"),
            )
        )

    def _remember_repl_result(self, result: REPLResult) -> REPLResult:
        self._last_repl_result = {
            "stdout_tail": result.stdout[-4000:],
            "stderr_tail": result.stderr[-4000:],
            "locals": result.locals,
            "execution_time": result.execution_time,
            "rlm_call_count": len(result.rlm_calls),
            "final_answer": result.final_answer,
        }
        return result

    def restart_sandbox(self) -> None:
        self._stop_poller()
        with contextlib.suppress(Exception):
            self.sandbox.process.delete_session(self.broker_session_id)
        self.sandbox.stop(timeout=self.timeout)
        self.sandbox.start(timeout=self.timeout)
        with contextlib.suppress(Exception):
            self.sandbox.wait_for_sandbox_start(timeout=self.timeout)
        with contextlib.suppress(Exception):
            self.sandbox.set_autostop_interval(self.auto_stop_interval)
        self._start_broker()
        self._reset_execution_state()

    def read_memory_cgroup(self) -> dict[str, int | None]:
        response = self.sandbox.process.exec(
            "sh -lc 'cat /sys/fs/cgroup/memory.current "
            "/sys/fs/cgroup/memory.peak 2>/dev/null'",
            timeout=30,
        )
        values = [line.strip() for line in str(response.result).splitlines()]
        current = int(values[0]) if len(values) >= 1 and values[0].isdigit() else None
        peak = int(values[1]) if len(values) >= 2 and values[1].isdigit() else None
        data = {"current_bytes": current, "peak_bytes": peak}
        baseline = getattr(self, "_baseline_memory_cgroup", None) or {}
        if baseline:
            data["baseline_current_bytes"] = baseline.get("current_bytes")
            data["baseline_peak_bytes"] = baseline.get("peak_bytes")
        return data

    def reset_memory_peak(self) -> bool:
        response = self.sandbox.process.exec(
            "sh -lc 'echo 0 > /sys/fs/cgroup/memory.peak'",
            timeout=30,
        )
        return getattr(response, "exit_code", 1) == 0

    def cleanup(self):
        if (
            self.execution_mode == "code_interpreter"
            and self._code_context is not None
            and self.sandbox is not None
        ):
            with contextlib.suppress(Exception):
                self.sandbox.code_interpreter.delete_context(self._code_context)
            self._code_context = None
        with contextlib.suppress(Exception):
            self._http.close()
        super().cleanup()
