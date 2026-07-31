"""Content-addressed read-only context store + lazy str view (RLM Lab, BR002-WO003).

Additive module — nothing here changes stock rlm behavior. Three pieces:

- ``ContextStore``: content-addressed (sha256 of the stored bytes), read-only,
  fixed-width **UTF-32-LE** text segments on disk + a per-store ``index.json``
  mapping logical names (task ids) to segment handles. A session references
  context via a **manifest** (ordered segment refs); WO003 exercises only
  single-segment manifests (replay never appends to the REPL context variable) —
  the append path is schema-defined here but not implemented (production
  concern, recorded as such in the WO).
- ``StoreRef``: a tiny picklable handle (store root + segment hash + char
  count). It is what flows through ``RLM.completion`` instead of the payload
  str, so neither the host nor the guest ever materializes the context
  (WO003 "handles end-to-end").
- ``RecorderStr``: a real ``str`` subclass that counts every Python-level str
  operation — the E1c op-census instrument (D-b). Zero compat risk (it *is* a
  str); C-level consumers bypass Python methods and are invisible by design
  (E3's LazyStr dry run is the detector for those).

Fixed width means O(1) char<->byte in both directions: char N = byte 4N
(FR002-CL001; naive UTF-32 first, per-segment narrowest width backlogged).
"""

from __future__ import annotations

import hashlib
import json
import mmap
import os
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

_CODEC = "utf-32-le"  # no BOM, fixed 4 bytes/char
_WIDTH = 4

# --------------------------------------------------------------------------- #
# Store
# --------------------------------------------------------------------------- #


class ContextStore:
    """Read-only, content-addressed UTF-32 segment store.

    Layout::

        <root>/segments/<sha256>.u32   raw UTF-32-LE payload (no BOM)
        <root>/index.json              {logical_name: {"handle", "chars", "bytes"}}

    A *manifest* is an ordered list of segment handles. WO003 uses
    single-segment manifests only; multi-segment (append/rope) is a schema, not
    an implementation.
    """

    def __init__(self, root: str | os.PathLike[str]):
        self.root = Path(root)
        self.segments_dir = self.root / "segments"

    # -- ingest (offline pre-ingest path, D-c) ------------------------------ #

    def ingest_text(self, text: str) -> str:
        """Encode + hash + write one segment; returns the handle (sha256 hex).

        Content-addressed on the *stored* bytes: identical text always hashes
        to the same segment, so re-ingest is a free dedup hit.
        """
        payload = text.encode(_CODEC)
        handle = hashlib.sha256(payload).hexdigest()
        seg = self.segments_dir / f"{handle}.u32"
        if not seg.exists():
            self.segments_dir.mkdir(parents=True, exist_ok=True)
            tmp = seg.with_suffix(".tmp")
            tmp.write_bytes(payload)
            os.replace(tmp, seg)  # atomic publish
        return handle

    def ingest_file(self, utf8_path: str | os.PathLike[str], name: str | None = None) -> dict:
        """Ingest one UTF-8 text file; returns an index entry (+ timing/bytes).

        The pre-ingest run's own wall time + disk bytes are the free per-MiB
        ingest datum (D-c) — logged by the caller, never inside replay rows.
        """
        p = Path(utf8_path)
        t0 = time.perf_counter()
        text = p.read_text(encoding="utf-8")
        handle = self.ingest_text(text)
        wall = time.perf_counter() - t0
        entry = {
            "handle": handle,
            "chars": len(text),
            "bytes": len(text) * _WIDTH,
            "src_bytes": p.stat().st_size,
            "ingest_s": round(wall, 3),
        }
        if name is not None:
            index = self.load_index()
            index[name] = entry
            (self.root / "index.json").write_text(json.dumps(index, indent=2))
        return entry

    def load_index(self) -> dict:
        idx = self.root / "index.json"
        if idx.exists():
            return json.loads(idx.read_text())
        return {}

    # -- open (replay path) -------------------------------------------------- #

    def segment_path(self, handle: str) -> Path:
        return self.segments_dir / f"{handle}.u32"

    def open(self, handle: str) -> LazyStr:
        """mmap the segment read-only and return the lazy str view."""
        seg = self.segment_path(handle)
        f = open(seg, "rb")
        try:
            mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
        finally:
            f.close()  # mmap holds its own reference
        return LazyStr(mm, handle=handle)

    def ref_for(self, name: str) -> StoreRef:
        entry = self.load_index()[name]
        return StoreRef(str(self.root), entry["handle"], entry["chars"])


@dataclass(frozen=True)
class StoreRef:
    """Picklable context handle — what crosses process/API boundaries.

    ``context_char_count`` is the duck-type hook ``QueryMetadata`` uses so the
    root-LM metadata (context length/type) is byte-identical to a str payload's.
    """

    store_root: str
    handle: str
    chars: int

    @property
    def context_char_count(self) -> int:
        return self.chars

    def open(self) -> LazyStr:
        return ContextStore(self.store_root).open(self.handle)

    @classmethod
    def from_index(cls, store_root: str | os.PathLike[str], name: str) -> StoreRef:
        return ContextStore(store_root).ref_for(name)


@dataclass(frozen=True)
class PathRef:
    """Plain path handle — the *cheap alternative* control (BR002-WO003 follow-up).

    The externalized lane changed two things at once versus legacy: it stopped
    materializing the payload host-side (**handle passing**) *and* it swapped the
    guest bind for a lazy mmap view (**the store**). Because the str façade
    escapes in 646/646 task rows, the guest ends up holding a full ``str`` in
    both lanes — so the measured per-session saving is plausibly attributable to
    handle passing alone. ``PathRef`` isolates that: it passes a path (no
    host-side copy) and lets the guest do the stock ``f.read()`` — no store, no
    UTF-32, no ``LazyStr``, no coercion shim, no escape hatch.

    It is deliberately a *marker type*, not a str: ``add_context`` must be able
    to tell "this is a handle" from "this is the payload".
    """

    path: str
    chars: int

    @property
    def context_char_count(self) -> int:
        # Same duck-type hook as StoreRef, so root-LM metadata is identical.
        return self.chars


# --------------------------------------------------------------------------- #
# LazyStr — the FR002 lazy view
# --------------------------------------------------------------------------- #

# Escape-hatch log (FR002: whole-object escapes full-materialize, logged and
# reported as façade cost). Module-level so the worker can snapshot per task.
_ESCAPES: Counter = Counter()
_LARGE_MATERIALIZE_CHARS = 1_000_000  # ops returning >= this many chars are logged


def escape_snapshot() -> dict:
    return dict(_ESCAPES)


def escape_reset() -> None:
    _ESCAPES.clear()


class LazyStr:
    """str-compatible lazy view over one mmap'd UTF-32-LE segment.

    Implements **exactly the E1c-census op set** natively/lazily —
    ``__getitem__`` (int/slice), ``find``, ``rfind``, ``__len__``, ``lower``,
    ``count``, ``split`` (census 2026-07-28: 46/46 tasks, polluted=0) — as
    byte-range reads; every value returned is a **real str**, so downstream
    library calls run natively on materialized slices. Everything outside the
    census (``in``, iteration, ``__str__``, any other str method via
    ``__getattr__``) full-materializes through the **logged escape hatch** —
    measured façade cost, never a crash, never a silent wrong answer. The two
    documented deviations: ``__eq__`` (chunked, lazy — Python's identity
    fallback would silently answer False for equal content) and ``__bool__``.
    Char N = byte 4N in both directions (fixed width) — slices and finds are
    O(range), not O(N).

    Deliberately NOT a ``str`` subclass: subclassing would materialize the full
    payload at construction, which defeats the design; ``isinstance(ctx, str)``
    misses and C-level exact-``str`` demands are expected boundary findings
    (WO003 risk register).
    """

    __slots__ = ("_mm", "_len", "_handle", "_materialized", "_int_hits")

    # A consumer that walks the view one character at a time (e.g.
    # difflib.SequenceMatcher, which accepts any sequence and therefore never
    # raises) is real façade cost that raises no exception and would otherwise
    # log nothing at all — the census would under-count by construction
    # (ABR 260731 finding). Crossing this many int-index reads logs it once.
    _INT_INDEX_STORM = 100_000

    def __init__(self, mm: mmap.mmap, handle: str = ""):
        if len(mm) % _WIDTH:
            raise ValueError("segment length is not a multiple of the char width")
        self._mm = mm
        self._len = len(mm) // _WIDTH
        self._handle = handle
        self._materialized: str | None = None
        self._int_hits = 0

    # -- internals ----------------------------------------------------------- #

    def _decode(self, start: int, end: int) -> str:
        """Decode chars [start, end) — faults in only the touched pages."""
        if end <= start:
            return ""
        return str(
            memoryview(self._mm)[start * _WIDTH : end * _WIDTH], _CODEC
        )

    def _norm(self, index: int | None, default: int) -> int:
        if index is None:
            return default
        if index < 0:
            index += self._len
        return min(max(index, 0), self._len)

    def _norm_start(self, index: int | None) -> int:
        """`start` as str treats it for an EMPTY needle: negatives wrap and
        floor at 0, but it is **not** clamped down to len — that is what makes
        `"abc".find("", 99)` return -1 rather than 3 (ABR 260731 R2-2)."""
        if index is None:
            return 0
        if index < 0:
            index += self._len
        return max(index, 0)

    def _log_large(self, op: str, chars: int) -> None:
        if chars >= _LARGE_MATERIALIZE_CHARS:
            _ESCAPES[f"large:{op}"] += 1

    def _materialize(self, op: str) -> str:
        """The escape hatch: logged whole-object materialization (cached —
        the session then honestly carries the full str in anon heap)."""
        _ESCAPES[f"escape:{op}"] += 1
        if self._materialized is None:
            self._materialized = self._decode(0, self._len)
        return self._materialized

    # -- core protocol -------------------------------------------------------- #

    def __len__(self) -> int:
        return self._len

    def __getitem__(self, key: int | slice) -> str:
        if isinstance(key, slice):
            start, stop, step = key.indices(self._len)
            if step == 1:
                out = self._decode(start, stop)
            else:
                out = "".join(
                    self._decode(i, i + 1) for i in range(start, stop, step)
                )
            self._log_large("__getitem__[slice]", len(out))
            return out
        index = key
        if index < 0:
            index += self._len
        if not 0 <= index < self._len:
            raise IndexError("string index out of range")
        self._int_hits += 1
        if self._int_hits == self._INT_INDEX_STORM:
            _ESCAPES["escape:int-index-storm"] += 1
        return self._decode(index, index + 1)

    def __bool__(self) -> bool:
        return self._len > 0

    # NOT in the E1c census -> logged escape hatch (FR002: only the audited op
    # set is implemented natively; anything else full-materializes, loudly).
    def __contains__(self, sub: str) -> bool:
        return sub in self._materialize("__contains__")

    def __iter__(self):
        return iter(self._materialize("__iter__"))

    # Protocol safety beyond the census (deliberate, documented deviation):
    # without __eq__, Python's identity fallback would answer `view == s` with
    # False for equal content — a SILENT wrong answer the escape log can never
    # see. Chunked compare, no materialization. __hash__ stays an escape.
    def __eq__(self, other: object) -> bool:
        if other is self:
            return True
        # View-to-view: without this, Python's identity fallback answers False
        # for two views over the same content — the same silent-wrong-answer
        # class this method exists to prevent (ABR 260731 finding).
        if isinstance(other, LazyStr):
            if other._len != self._len:
                return False
            if other._handle and other._handle == self._handle:
                return True
            chunk = 1 << 20
            for start in range(0, self._len, chunk):
                end = min(start + chunk, self._len)
                if self._decode(start, end) != other._decode(start, end):
                    return False
            return True
        if not isinstance(other, str):
            return NotImplemented
        if len(other) != self._len:
            return False
        chunk = 1 << 20
        for start in range(0, self._len, chunk):
            end = min(start + chunk, self._len)
            if self._decode(start, end) != other[start:end]:
                return False
        return True

    def __ne__(self, other: object) -> bool:
        result = self.__eq__(other)
        return NotImplemented if result is NotImplemented else not result

    def __hash__(self) -> int:
        return hash(self._materialize("__hash__"))

    def __str__(self) -> str:
        return self._materialize("__str__")

    def __repr__(self) -> str:
        return repr(self._materialize("__repr__"))

    def __add__(self, other: str) -> str:
        return self._materialize("__add__") + other

    def __radd__(self, other: str) -> str:
        return other + self._materialize("__radd__")

    def __getattr__(self, name: str):
        # Unknown str API → logged escape hatch, then delegate natively.
        if name.startswith("_"):
            raise AttributeError(name)
        target = self._materialize(f"getattr:{name}")
        return getattr(target, name)

    # -- census-backed op set ------------------------------------------------- #

    def find(self, sub: str, start: int | None = None, end: int | None = None) -> int:
        lo, hi = self._norm(start, 0), self._norm(end, self._len)
        if not sub:
            raw = self._norm_start(start)
            return raw if raw <= hi else -1
        needle = sub.encode(_CODEC)
        pos = lo * _WIDTH
        limit = hi * _WIDTH
        while True:
            hit = self._mm.find(needle, pos, limit)
            if hit == -1:
                return -1
            if hit % _WIDTH == 0:
                return hit // _WIDTH
            pos = hit + 1  # unaligned byte coincidence — keep scanning

    def rfind(self, sub: str, start: int | None = None, end: int | None = None) -> int:
        lo, hi = self._norm(start, 0), self._norm(end, self._len)
        if not sub:
            raw = self._norm_start(start)
            return hi if raw <= hi else -1
        needle = sub.encode(_CODEC)
        limit = hi * _WIDTH
        pos = lo * _WIDTH
        while True:
            hit = self._mm.rfind(needle, pos, limit)
            if hit == -1:
                return -1
            if hit % _WIDTH == 0:
                return hit // _WIDTH
            limit = hit + len(needle) - 1  # unaligned — search strictly before

    def count(self, sub: str, start: int | None = None, end: int | None = None) -> int:
        lo, hi = self._norm(start, 0), self._norm(end, self._len)
        if not sub:
            # str yields 0 for an inverted or past-the-end range, not 1.
            raw = self._norm_start(start)
            return hi - raw + 1 if hi >= raw else 0
        n = 0
        pos = lo
        while True:
            hit = self.find(sub, pos, hi)
            if hit == -1:
                return n
            n += 1
            pos = hit + len(sub)

    def lower(self) -> str:
        out = self._decode(0, self._len).lower()
        self._log_large("lower", len(out))
        return out

    def split(self, sep: str | None = None, maxsplit: int = -1) -> list[str]:
        out = self._decode(0, self._len).split(sep, maxsplit)
        self._log_large("split", self._len)
        return out

    @property
    def context_char_count(self) -> int:
        # Duck-type hook shared with StoreRef (QueryMetadata compatibility).
        return self._len


# --------------------------------------------------------------------------- #
# C-level boundary: coercing import shim
# --------------------------------------------------------------------------- #

# Modules whose C implementations demand an exact ``str`` (PyUnicode_Check) and
# therefore cannot consume a lazy view. FR002 ratifies the behavior — "C-level
# exact-`str` demands full-materialize as an escape hatch, logged and reported
# as façade cost" — this shim is the mechanism that fires it instead of letting
# the guest see a TypeError. Non-LazyStr arguments pass through untouched, so
# the shim is an identity transform for every other call.
_C_LEVEL_MODULES = frozenset(
    {"re", "json", "difflib", "textwrap", "unicodedata", "csv", "base64",
     "hashlib", "html", "string", "shlex", "ast"}
)


def _coerce(value, where: str):
    if isinstance(value, LazyStr):
        return value._materialize(f"c-level:{where}")
    return value


class _CoercingProxy:
    """Materializes LazyStr arguments (logged) before a C-level call."""

    __slots__ = ("_wrapped", "_where")

    def __init__(self, wrapped, where: str):
        object.__setattr__(self, "_wrapped", wrapped)
        object.__setattr__(self, "_where", where)

    def __getattr__(self, name: str):
        attr = getattr(object.__getattribute__(self, "_wrapped"), name)
        where = f"{object.__getattribute__(self, '_where')}.{name}"
        # Classes pass through UNWRAPPED, deliberately. Wrapping them coerces
        # constructor args, but a proxy is not a class: `except
        # json.JSONDecodeError` becomes an uncatchable TypeError,
        # `isinstance(node, ast.Assign)` silently flips True→False, and
        # subclassing breaks. That trade was measured and rejected — the
        # class-constructor path has zero occurrences in the corpus, while
        # except/isinstance/subclass are ordinary. Class-level consumers remain
        # a known boundary; the `__getitem__[int]`-storm counter is the defense
        # that costs nothing (ABR 260731 R2-1).
        if callable(attr) and not isinstance(attr, type):
            return _coercing_call(attr, where)
        return attr

    def __dir__(self):
        return dir(object.__getattribute__(self, "_wrapped"))

    def __repr__(self) -> str:
        return repr(object.__getattribute__(self, "_wrapped"))


def _coercing_call(func, where: str):
    def wrapper(*args, **kwargs):
        args = tuple(_coerce(a, where) for a in args)
        kwargs = {k: _coerce(v, where) for k, v in kwargs.items()}
        result = func(*args, **kwargs)
        # Compiled patterns carry their own C-level methods (p.finditer(ctx)),
        # so the proxy has to ride along with them.
        if type(result).__name__ in ("Pattern", "Scanner"):
            return _CoercingProxy(result, f"{where}()")
        return result

    wrapper.__name__ = getattr(func, "__name__", "wrapped")
    wrapper.__doc__ = getattr(func, "__doc__", None)
    return wrapper


def make_coercing_import(real_import=None):
    """Return an ``__import__`` that wraps C-level text modules (guest builtins)."""
    real = real_import or __import__

    def _import(name, globals=None, locals=None, fromlist=(), level=0):  # noqa: A002
        module = real(name, globals, locals, fromlist, level)
        root = name.split(".", 1)[0]
        if root in _C_LEVEL_MODULES:
            return _CoercingProxy(module, root)
        return module

    return _import


# --------------------------------------------------------------------------- #
# E1c census instrument (D-b): RecorderStr
# --------------------------------------------------------------------------- #

_RECORD: Counter = Counter()

_STR_METHODS = [
    # dunders (Python-level slots; C fast paths bypass by design)
    "__len__", "__contains__", "__iter__", "__eq__", "__ne__",
    "__lt__", "__le__", "__gt__", "__ge__", "__add__", "__mul__",
    "__mod__", "__hash__", "__str__", "__repr__", "__format__",
    # public API
    "capitalize", "casefold", "center", "count", "encode", "endswith",
    "expandtabs", "find", "format", "format_map", "index", "isalnum",
    "isalpha", "isascii", "isdecimal", "isdigit", "isidentifier", "islower",
    "isnumeric", "isprintable", "isspace", "istitle", "isupper", "join",
    "ljust", "lower", "lstrip", "partition", "removeprefix", "removesuffix",
    "replace", "rfind", "rindex", "rjust", "rpartition", "rsplit", "split",
    "splitlines", "startswith", "strip", "swapcase", "title", "translate",
    "upper", "zfill",
]


def _recorder_namespace() -> dict:
    ns: dict = {"__slots__": ()}

    def _wrap(name: str):
        orig = getattr(str, name)

        def method(self, *args, **kwargs):
            _RECORD[name] += 1
            return orig(self, *args, **kwargs)

        method.__name__ = name
        return method

    for _name in _STR_METHODS:
        ns[_name] = _wrap(_name)

    # __getitem__ split by arg kind — LazyStr's two distinct access paths
    _orig_getitem = str.__getitem__

    def __getitem__(self, key):  # noqa: N807
        _RECORD["__getitem__[slice]" if isinstance(key, slice) else "__getitem__[int]"] += 1
        return _orig_getitem(self, key)

    ns["__getitem__"] = __getitem__

    @classmethod
    def snapshot(cls) -> dict:
        return dict(_RECORD)

    @classmethod
    def reset(cls) -> None:
        _RECORD.clear()

    ns["snapshot"] = snapshot
    ns["reset"] = reset
    return ns


RecorderStr = type("RecorderStr", (str,), _recorder_namespace())
