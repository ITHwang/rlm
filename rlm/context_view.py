"""Lazy context view over the ORIGINAL UTF-8 file (RLM Lab, BR002-WO009).

This replaces ``context_store.py``'s UTF-32 ContextStore. The governing rule is
one line: **do not make new bytes.** WO003 built a store, wrote the corpus into
it as UTF-32, mapped that, and measured the store's contribution to H6(a) as
zero — while the UTF-32 encoding simultaneously blocked the only mechanism that
could have rescued H6(a), because ``re`` will scan any buffer-protocol object
for a ``bytes`` pattern and fixed-width padding is data to a byte regex.

So: map the original prompt file, and pay for a small index instead of a 4x
rewrite of the corpus.

Layers (modality generalisation, WO009 D-h). L1 and L4 are modality-agnostic and
written once; L2's *interface* is shared and its implementation is per-modality;
only L3 is written fresh per modality::

    L4  EscapeHatch   fires when a consumer demands a materialised object
    L3  TextView      slice / find / rfind / len / the re protocol
    L2  Coords        Protocol{to_byte, to_native, length}
                      Utf8Coords (checkpoints) | IdentityCoords (identity)
    L1  MappedBytes   one file mmap'd, byte-range reads

Search is served in three tiers, each strictly better than the next, so the
worst case is exactly today's behaviour and the change cannot regress:

    tier 1  translate the str pattern to an equivalent BYTES pattern and scan
            the mapping in place                          -> +0.0 MiB anon
    tier 2  decode a bounded window and run the ORIGINAL pattern on it
            (exact by construction, no translator trusted) -> +window
    tier 3  materialise the whole context                  -> +125 MiB (WO003)
"""

from __future__ import annotations

import mmap
import os
import re

try:  # CPython >= 3.11 exposes the sre internals here
    from re import _constants as _sre_c
    from re import _parser as _sre_p
except ImportError:  # pragma: no cover - older interpreters
    import sre_constants as _sre_c  # type: ignore[no-redef]
    import sre_parse as _sre_p  # type: ignore[no-redef]
from array import array
from collections import Counter
from pathlib import Path
from typing import Protocol, runtime_checkable

__all__ = [
    "Coords",
    "IdentityCoords",
    "Utf8Coords",
    "MappedBytes",
    "TextView",
    "PathRef",
    "ViewRef",
    "translate_pattern",
    "utf8_char_count",
    "escape_snapshot",
    "escape_reset",
    "search_tier_snapshot",
    "make_coercing_import",
]

# --------------------------------------------------------------------------- #
# L4 — escape / tier accounting (modality-agnostic)
# --------------------------------------------------------------------------- #
_ESCAPES: Counter = Counter()
_TIERS: Counter = Counter()


def _log_escape(where: str) -> None:
    _ESCAPES[where] += 1


def _log_tier(tier: str, api: str) -> None:
    _TIERS[f"{tier}:{api}"] += 1


def escape_snapshot() -> dict:
    return dict(_ESCAPES)


def escape_reset() -> None:
    _ESCAPES.clear()
    _TIERS.clear()


def search_tier_snapshot() -> dict:
    return dict(_TIERS)


# --------------------------------------------------------------------------- #
# L2 — coordinates
# --------------------------------------------------------------------------- #
@runtime_checkable
class Coords(Protocol):
    """Translate a modality's natural coordinate to/from a byte offset.

    Text is the hard case (UTF-8 is variable width); binary modalities are the
    identity, which is why the interface exists at all — an image or an audio
    buffer needs ``IdentityCoords`` and nothing else.
    """

    def to_byte(self, pos: int) -> int: ...
    def to_native(self, byte_off: int) -> int: ...
    def length(self) -> int: ...


class IdentityCoords:
    """Binary modalities: the natural coordinate *is* the byte offset.

    This is the proof that L2 generalises — a modality with no character notion
    needs three one-line methods and nothing else. No second modality is built
    in this WO (D-h); only the boundary is.
    """

    __slots__ = ("_n",)

    def __init__(self, nbytes: int):
        self._n = nbytes

    def to_byte(self, pos: int) -> int:
        return pos

    def to_native(self, byte_off: int) -> int:
        return byte_off

    def length(self) -> int:
        return self._n


# Counting codepoints is counting bytes that are NOT continuation bytes. Doing
# that with a Python loop costs ~4 s per 31 MiB prompt; `translate` + `count`
# are two C passes and cost milliseconds. The table maps every continuation
# byte (10xxxxxx) to 1 and everything else to 0.
_CONT_TABLE = bytes(1 if (b & 0xC0) == 0x80 else 0 for b in range(256))


def _count_chars(chunk: bytes) -> int:
    return len(chunk) - chunk.translate(_CONT_TABLE).count(1)


class Utf8Coords:
    """Checkpoint index over a UTF-8 file — char offset <-> byte offset.

    Two 32-bit arrays, both directions O(1) plus a bounded in-block scan::

        by_block[i] = chars before byte  i*K   ->  byte->char is  b // K
        by_chunk[j] = byte of char       j*K   ->  char->byte is  c // K

    One array would leave ``char->byte`` needing a binary search, and that is
    the hotter direction (WO003 census: ``slice`` 71,883 calls vs
    ``find``+``rfind`` 14,186). The second array costs ~31 KB per file and
    removes the search.

    MEASURED, not estimated: 71.5 KB per file and **3.21 MiB** over the 46-file
    corpus — 0.219% of the 1.429 GiB it indexes, against the UTF-32 store's
    5.69 GiB. (The round-1 figure of 63 KB/2.9 MB was an arithmetic estimate
    that omitted ``_ascii`` and assumed a 4-byte word the code did not use.)

    ``_ascii`` marks pure-ASCII blocks, where char and byte offsets move
    together — 99.709% of this corpus, so the bounded scan almost never runs
    and whole blocks are crossed by arithmetic.
    """

    __slots__ = (
        "_k", "_by_block", "_by_chunk", "_ascii", "_nbytes", "_nchars", "_data",
    )

    K = 4096

    def __init__(self, data: memoryview | bytes, block: int = K):
        self._k = block
        self._data = data
        nbytes = len(data)
        self._nbytes = nbytes

        # Pass 1 — one entry per K-BYTE block: chars seen before it, and whether
        # the block is pure ASCII (no scan is ever needed inside one).
        # 'I' is 4 bytes; 'L' is 8 on LP64 and would silently double an index
        # this file's own docstring describes as 32-bit. Both arrays hold
        # offsets bounded by the file size, so 32 bits is enough below 4 GiB —
        # above it, pay the wider word rather than wrap.
        code = "I" if nbytes < (1 << 32) else "L"
        by_block = array(code)
        ascii_flags = bytearray()
        chars = 0
        pos = 0
        while pos < nbytes:
            end = min(pos + block, nbytes)
            by_block.append(chars)
            chunk = bytes(data[pos:end])
            if chunk.isascii():
                ascii_flags.append(1)
                chars += end - pos
            else:
                ascii_flags.append(0)
                chars += _count_chars(chunk)
            pos = end
        by_block.append(chars)
        ascii_flags.append(1)
        self._nchars = chars
        self._by_block = by_block
        self._ascii = bytes(ascii_flags)

        # Pass 2 — one entry per K-CHAR chunk: the byte it starts at. The block
        # cursor only moves forward, so the whole pass is linear.
        by_chunk = array(code)
        i = 0
        n_blocks = max(1, len(by_block) - 1)
        target = 0
        while target * block <= chars:
            want = target * block
            while i + 1 < n_blocks and by_block[i + 1] <= want:
                i += 1
            by_chunk.append(self._walk(i * block, by_block[i], want))
            target += 1
        self._by_chunk = by_chunk

    def _walk(self, byte_at: int, char_at: int, want_char: int) -> int:
        """Advance from a known (byte, char) pair to ``want_char``.

        Bounded by one block per call because every caller starts from the
        nearest checkpoint, and ASCII blocks are crossed by arithmetic.
        """
        data = self._data
        n = self._nbytes
        b, c = byte_at, char_at
        # A K-byte block boundary is an arbitrary byte offset and can land in
        # the MIDDLE of a character. ``by_block`` counts characters that have
        # *started*, so the char at ``char_at`` begins at the next lead byte —
        # skip to it without counting, or every checkpoint inside a non-ASCII
        # block is one character early.
        while b < n and (data[b] & 0xC0) == 0x80:
            b += 1
        while c < want_char and b < n:
            blk = b // self._k
            if blk < len(self._ascii) and self._ascii[blk]:
                room = min(self._k - (b % self._k), want_char - c, n - b)
                if room <= 0:
                    break
                b += room
                c += room
                continue
            b += 1
            while b < n and (data[b] & 0xC0) == 0x80:
                b += 1
            c += 1
        return b

    def length(self) -> int:
        return self._nchars

    def to_native(self, byte_off: int) -> int:
        """byte -> char. One division, one array read, then a bounded scan the
        ASCII flag skips entirely for 99.7% of this corpus's blocks."""
        if byte_off <= 0:
            return 0
        if byte_off >= self._nbytes:
            return self._nchars
        i = byte_off // self._k
        base = self._by_block[i]
        start = i * self._k
        if self._ascii[i]:
            return base + (byte_off - start)
        return base + _count_chars(bytes(self._data[start:byte_off]))

    def to_byte(self, pos: int) -> int:
        """char -> byte. Same shape, the other array."""
        if pos <= 0:
            return 0
        if pos >= self._nchars:
            return self._nbytes
        j = min(pos // self._k, len(self._by_chunk) - 1)
        return self._walk(self._by_chunk[j], j * self._k, pos)


# --------------------------------------------------------------------------- #
# L1 — mapped bytes (modality-agnostic)
# --------------------------------------------------------------------------- #
class MappedBytes:
    """One file, mmap'd read-only. Nothing is read until a page is touched, and
    every page that is touched is clean file-backed page cache — evictable for
    free, which is the whole point (H6(b))."""

    __slots__ = ("path", "_fd", "_mm", "_view")

    def __init__(self, path: str | os.PathLike[str]):
        self.path = str(path)
        self._fd = os.open(self.path, os.O_RDONLY)
        try:
            # mmap refuses a zero-length file outright, so an empty context —
            # legal input, and what a truncated fixture looks like — crashed the
            # constructor instead of yielding an empty view.
            if os.fstat(self._fd).st_size == 0:
                self._mm = b""
            else:
                self._mm = mmap.mmap(self._fd, 0, access=mmap.ACCESS_READ)
        except Exception:
            os.close(self._fd)
            raise
        self._view = memoryview(self._mm)

    def __len__(self) -> int:
        return len(self._mm)

    @property
    def buffer(self) -> mmap.mmap:
        """The object handed to ``re`` for a bytes-pattern scan (+0.0 MiB)."""
        return self._mm

    def __getitem__(self, item):
        return self._view[item]

    def slice_bytes(self, start: int, end: int) -> bytes:
        return bytes(self._view[start:end])

    def close(self) -> None:
        try:
            self._view.release()
        finally:
            if not isinstance(self._mm, bytes):
                self._mm.close()
            os.close(self._fd)


# --------------------------------------------------------------------------- #
# tier 1 — pattern translation (str pattern -> equivalent bytes pattern)
# --------------------------------------------------------------------------- #
# One UTF-8 codepoint as bytes. `.` in bytes mode matches ONE BYTE and would
# split a character, so it has to be replaced — but *how* it is replaced decides
# whether tier 1 is usable at all.
#
# The obvious replacement is the textbook 9-branch alternation over the UTF-8
# byte ranges. It is exact, and it is a trap: under a quantifier (`.{0,200}` is
# ordinary in this corpus) the engine backtracks across nine branches per
# repetition and a 31 MiB scan stops finishing. That was measured, not guessed —
# a task ran 24 minutes at 100% CPU and py-spy put the whole stack in this loop.
#
# The cheap form below is exact **on valid UTF-8**, which the subject always is:
# a codepoint is one non-continuation byte followed by the continuation bytes
# that belong to it. Two tokens, no alternation, no backtracking.
#
# The group is not decoration. A quantifier binds to the *last token*, so a bare
# two-token expansion makes `a.?b` compile as `a<lead><cont>*?b` — the `?` turns
# the continuation repeat lazy instead of making the character optional, and
# `a.+b` becomes `...*+`, a POSSESSIVE repeat on 3.11+. Both compile, so the
# re.compile guard never fires; both silently return wrong answers.
# The group must be ATOMIC, not merely non-capturing. Grouping alone fixes the
# binding but hands the engine a nested quantifier — `.{0,200}` becomes
# `(?:X Y*){0,200}`, and it can then redistribute continuation bytes between the
# inner `*` and the outer repeat. That is the catastrophic case described above,
# and it is worse than a slow scan: round 1 got away with it only because the
# ungrouped form failed to compile at all (`*{0,200}` is a "multiple repeat"
# error) and silently fell through to tier 2. Fixing the binding re-enabled the
# pattern on tier 1 and a real ladder row timed out at 1278 s against a 900 s
# limit — a cost no correctness suite could see.
#
# UTF-8 is self-synchronising, so the greedy run of continuation bytes after a
# lead byte is the ONLY correct parse: there is never a reason to backtrack into
# it. `(?>...)` says exactly that, and makes the repeat linear.
_UTF8_CHAR = rb"(?>[^\x80-\xbf][\x80-\xbf]*)"
_UTF8_CHAR_NO_NL = rb"(?>[^\x80-\xbf\n][\x80-\xbf]*)"

# The same rule governs every emitted token: anything that is not a single byte
# or an already-grouped alternation has to be atomic before a quantifier can
# reach it. A literal `é` encodes to TWO bytes, so `é+` would otherwise compile
# as `\xc3\xa9+` — one lead byte and a repeated continuation byte.
def _atom(token: bytes) -> bytes:
    """Make `token` safe to carry a quantifier."""
    if len(token) == 1:
        return token
    if token.startswith((b"(?:", b"(?>")) and token.endswith(b")"):
        return token
    if token.startswith(b"[") and token.endswith(b"]") and b"]" not in token[1:-1]:
        return token
    if len(token) == 2 and token[0:1] == b"\\":
        return token  # an escaped single byte, e.g. \n or \.
    return b"(?:" + token + b")"

# CPython's `re` uses SIMPLE case folding. Of the 12 non-ASCII codepoints that
# fold into ASCII, **four** participate in an ASCII-pattern `re.I` match — the
# round-1 set omitted U+0130 and shipped. Enumerated over the whole of Unicode
# by a test rather than asserted here (BR002-WO009 E3):
#   U+0130 CAPITAL I WITH DOT  <-> i/I     U+0131 DOTLESS I <-> i/I
#   U+017F LONG S              <-> s/S     U+212A KELVIN SIGN <-> k/K
# Expanding these inline makes `re.I` EXACTLY translatable, which matters
# because a precondition-based route was tried first and died: 0 of 46 corpus
# files are free of ASCII-folding codepoints.
_FOLD_EXTRA = {
    "i": ("\u0131", "\u0130"),  # DOTLESS I, CAPITAL I WITH DOT ABOVE
    "s": ("\u017f",),            # LONG S
    "k": ("\u212a",),            # KELVIN SIGN
}


def _lit_bytes(ch: str, ignorecase: bool, fold_unicode: bool = True) -> bytes:
    """One literal character as an exact bytes sub-pattern.

    Two traps, both found by review after the first version shipped:

    * `str.upper()` is FULL case mapping while `re` folding is SIMPLE, so
      `\u00df`.upper() is "SS" and `\ufb01`.upper() is "FI". Emitting those as
      alternatives makes tier 1 match where CPython does not — a spurious hit,
      which is worse than a miss. Simple folding never changes length, so
      length is the filter.
    * The fold set is FOUR codepoints, not three: U+0130 was missing, so
      `re.search("istanbul", "\u0130stanbul", re.I)` matched in CPython and not
      on tier 1.
    """
    if not ignorecase:
        return _atom(re.escape(ch.encode("utf-8")))
    if not ch.isascii():
        # Case folding is not a function of `lower`/`upper`/`casefold` for
        # non-ASCII: `\u00df` folds with `\u1e9e`, `\u00c5`/`\u00e5` with
        # U+212B, the Greek block has several partners each, and titlecase
        # digraphs have three. Enumerating the true partner set means indexing
        # the whole of Unicode; refusing costs one tier-2 window and is exact.
        # A miss is still a wrong answer, so this is not a stylistic choice.
        raise Untranslatable("non-ASCII literal under re.I")
    variants = {ch} | {
        v for v in (ch.lower(), ch.upper(), ch.casefold()) if len(v) == 1
    }
    if fold_unicode:
        variants.update(_FOLD_EXTRA.get(ch.lower(), ()))
    encoded = sorted({re.escape(v.encode("utf-8")) for v in variants if v})
    if len(encoded) == 1:
        return _atom(encoded[0])
    return b"(?:" + b"|".join(encoded) + b")"


_CLASS_MAP = {
    "d": (rb"[0-9]", True),  # (bytes form, needs no non-ASCII members?)
    "D": (None, False),
    "w": (None, False),
    "W": (None, False),
    "s": (None, False),
    "S": (None, False),
    # `\b`/`\B` are NOT translatable and must never be copied through. In str
    # mode the boundary is Unicode-aware; in bytes mode it is ASCII-only, so
    # every UTF-8 continuation byte reads as a non-word character and a boundary
    # appears in the middle of any non-ASCII word: `\bcat\b` on "naïcat and cat"
    # yields 2 matches against str's 1. Expressing it exactly would need a
    # variable-width lookbehind, which does not exist — so refuse.
    "b": (None, False),
    "B": (None, False),
    "A": (rb"\A", True),
    "Z": (rb"\Z", True),
    "n": (rb"\n", True),
    "t": (rb"\t", True),
    "r": (rb"\r", True),
    "f": (rb"\f", True),
    "v": (rb"\v", True),
}


def _class_alternation(str_class: str, ascii_part: bytes) -> bytes:
    """Exact bytes form of a Unicode-aware class: the ASCII members as a byte
    class, plus an explicit alternation of every non-ASCII member's UTF-8."""
    rx = re.compile(str_class)
    alts = [
        re.escape(chr(cp).encode("utf-8"))
        for cp in range(0x80, 0x110000)
        if rx.fullmatch(chr(cp))
    ]
    if not alts:
        return ascii_part
    return b"(?:" + ascii_part + b"|" + b"|".join(alts) + b")"


_LAZY_CLASS_CACHE: dict[str, bytes] = {}


def _unicode_class(name: str) -> bytes | None:
    if name in _LAZY_CLASS_CACHE:
        return _LAZY_CLASS_CACHE[name]
    spec = {
        "s": (r"\s", rb"[ \t\n\r\f\v]"),
        "d": (r"\d", rb"[0-9]"),
        "w": (r"\w", rb"[A-Za-z0-9_]"),
    }.get(name)
    if spec is None:
        return None
    out = _class_alternation(*spec)
    _LAZY_CLASS_CACHE[name] = out
    return out


# Flags whose bytes-mode consequence has been reasoned about. IGNORECASE and
# DOTALL are handled by the translation, MULTILINE is carried as `(?m)`, VERBOSE
# is refused outright. UNICODE is the str-mode default and a no-op. ASCII is NOT
# merely a narrowing of `\w \d \s \b` — it also narrows IGNORECASE, which the
# round-1 note got wrong; it is handled by `fold_unicode` below.
# Anything else — LOCALE, DEBUG, a future flag — refuses, because a flag that is
# silently dropped is exactly the silent-wrong-answer class this tier fears.
_KNOWN_FLAGS = (
    re.IGNORECASE | re.DOTALL | re.MULTILINE | re.VERBOSE | re.UNICODE | re.ASCII
)


# Constructs whose meaning is defined against the WHOLE subject. A bounded
# window is a different subject, so the engine evaluates them against the
# window's edges instead of the file's: `^` fires at every window start, `\b`
# fires wherever a window happens to begin mid-word, and a lookaround sees
# truncated context. None of that is repairable by growing the window, so a
# pattern carrying any of them leaves tier 2 for tier 3, where the subject is
# the whole context and the answer is exactly today's.
def _has_quantified_dot(pattern: str) -> bool:
    """Is any `.` in reach of a repeat operator, directly or through a group?"""
    i, n, in_class, dot_at = 0, len(pattern), False, None
    while i < n:
        ch = pattern[i]
        if ch == "\\":
            i += 2
            continue
        if in_class:
            in_class = ch != "]"
        elif ch == "[":
            in_class = True
        elif ch == ".":
            dot_at = i
        elif ch == "(" and pattern.startswith("(?", i):
            i += 2  # a group flavour's `?` is not a repeat operator
            continue
        elif ch in "*+?{" and dot_at is not None:
            return True
        i += 1
    return False


def _needs_full_subject(pattern: str) -> bool:
    i, n = 0, len(pattern)
    in_class = False
    while i < n:
        ch = pattern[i]
        if ch == "\\":
            if not in_class and i + 1 < n and pattern[i + 1] in "bBAZ":
                return True
            i += 2
            continue
        if in_class:
            in_class = ch != "]"
            i += 1
            continue
        if ch == "[":
            in_class = True
        elif ch in "^$":
            return True
        elif ch == "(" and pattern.startswith(("(?=", "(?!", "(?<=", "(?<!"), i):
            return True
        i += 1
    return False


# --------------------------------------------------------------------------- #
# tier-2 admission — the tail-position partition (BR002-FR006-CL004's successor
# item, built 2026-08-03 after ABR rounds 10-11 measured the corpus VIOLATING
# the unpartitioned tier's precondition)
# --------------------------------------------------------------------------- #
# The window fails in exactly one place: a match longer than the overlap can be
# TRUNCATED by the window edge, and a truncated attempt that *fails* leaves no
# trace — "no match" and "no match because it broke" are the same observation
# (five detection schemes died to counterexamples in ABR rounds 4-8). But the
# failure is only untraceable when something consuming FOLLOWS the long part:
# `A.*B` truncated loses `B` and evaporates. When every unbounded repeat sits
# in TAIL position (`A.*` — nothing consuming follows on any alternation path),
# a truncated attempt necessarily ends at the window's last character, which is
# the `m.end() == len(text)` signal `_windowed_finditer` already computes and
# escalates on. So the partition is the detectability boundary itself:
#
#   COMPOSITE max width <= overlap           -> tier 2 (the overlap argument)
#   wide, and shaped as NARROW PREFIX +
#     ONE width-1-body tail repeat, where
#     prefix and prefix+engaged-minimum
#     both fit the overlap                    -> tier 2 (edge signal catches it)
#   anything else                             -> tier 3 (exact by construction)
#
# The rule is deliberately COMPOSITE — ABR 260803-2100 demonstrated silent
# divergences twice, and both times the per-unit version of a guard was the
# hole: round 1 with single units (multi-char bodies retreating past the
# alarm; an optional hiding a wide minimum), round 2 with compositions of
# narrow pieces (a chain of narrow repeats + trailing consumer losing a match;
# a byte-inflated prefix starving an engaged optional into a wrong span). All
# widths are summed and compared in characters against overlap/4, so the ×4
# UTF-8 bound applies to the SUM in bytes — conservative in the safe
# direction, as is every unknown construct below.
_WIDE = 1 << 62


def _sat(n: int) -> int:
    return _WIDE if n > _WIDE else n


def _node_widths(item, gw: dict) -> tuple[int, int]:
    """(worst-alternation-path min, max) width of one parsed node, in chars.

    `gw` accumulates per-group widths for backreferences. Overestimating is
    safe (costs a tier-3 materialisation, which is exact); underestimating is
    not — so every unrecognised op is (WIDE, WIDE).
    """
    op, av = item
    if op in (_sre_c.LITERAL, _sre_c.NOT_LITERAL, _sre_c.ANY, _sre_c.IN):
        return (1, 1)
    if op is _sre_c.AT:
        return (0, 0)
    if op in (_sre_c.ASSERT, _sre_c.ASSERT_NOT):
        return (0, 0)  # zero-width; the lookaround itself is refused earlier
    if op is _sre_c.SUBPATTERN:
        gid = av[0]
        lo, hi = _seq_widths(av[3].data, gw)
        if gid:
            gw[gid] = (lo, hi)
        return (lo, hi)
    if op is getattr(_sre_c, "ATOMIC_GROUP", None):
        return _seq_widths(av.data, gw)
    if op is _sre_c.BRANCH:
        # min is the MAX over branches: the admission question is "can the
        # worst path still fail wide", so the pessimistic path is the bound.
        los, his = zip(*(_seq_widths(b.data, gw) for b in av[1]), strict=True)
        return (max(los), max(his))
    if op in (
        _sre_c.MAX_REPEAT,
        _sre_c.MIN_REPEAT,
        getattr(_sre_c, "POSSESSIVE_REPEAT", _sre_c.MAX_REPEAT),
    ):
        mn, mx, sub = av
        lo, hi = _seq_widths(sub.data, gw)
        top = _WIDE if (mx == _sre_c.MAXREPEAT and hi > 0) else _sat(int(mx) * hi)
        return (_sat(int(mn) * lo), top)
    if op is _sre_c.GROUPREF:
        return gw.get(av, (_WIDE, _WIDE))
    if op is _sre_c.GROUPREF_EXISTS:
        gid, yes, no = av
        ylo, yhi = _seq_widths(yes.data, gw)
        nlo, nhi = _seq_widths(no.data, gw) if no is not None else (0, 0)
        return (max(ylo, nlo), max(yhi, nhi))
    return (_WIDE, _WIDE)


def _seq_widths(items, gw: dict) -> tuple[int, int]:
    lo = hi = 0
    for it in items:
        item_lo, item_hi = _node_widths(it, gw)
        lo, hi = _sat(lo + item_lo), _sat(hi + item_hi)
    return (lo, hi)


def _admissible(items, prefix_hi: int, thresh: int, gw: dict) -> bool:
    r"""ABR 260803-2100 round 2: admission is judged COMPOSITELY, or not at all.

    Round 1's defects were single units and the round-1 fixes were per-unit
    guards, so round 2 composed narrow pieces into the same traceless
    failures: a chain of individually-narrow repeats with a trailing consumer
    lost a match entirely (`Sa{0,128}b{0,128}c{0,128}d{0,128}\d{0,128}Z`),
    and a byte-inflated prefix starved an engaged optional into a wrong short
    span (`S𐍈{65000}(?:\d{60000})?`). The closing rule is the
    conservative form the review names: a pattern whose worst-case width
    exceeds the overlap is admitted ONLY as

        NARROW PREFIX  +  ONE width-1-body tail repeat,

    where the prefix's summed worst-case width fits ``thresh`` (chars vs
    ``overlap // 4``, so the ×4 UTF-8 bound applies to the SUM, in bytes) and
    the prefix plus the tail's engaged MINIMUM also fits. Then: the prefix
    always fits the worst runway; the tail consumes one char at a time, so a
    cut necessarily ends the match AT the window edge — the alarm's signal.
    Groups/branches unwrap only in tail position; everything else wide is out.
    """
    total = _sat(prefix_hi + _seq_widths(items, gw)[1])
    if total <= thresh:
        return True  # composite-narrow: every possible match fits the overlap
    if not items:
        return False
    *init, last = items
    init_hi = _sat(prefix_hi + _seq_widths(init, gw)[1])
    op, av = last
    if op is _sre_c.SUBPATTERN:
        return _admissible(av[3].data, init_hi, thresh, gw)
    if op is getattr(_sre_c, "ATOMIC_GROUP", None):
        return _admissible(av.data, init_hi, thresh, gw)
    if op is _sre_c.BRANCH:
        return all(_admissible(b.data, init_hi, thresh, gw) for b in av[1])
    if op in (
        _sre_c.MAX_REPEAT,
        _sre_c.MIN_REPEAT,
        getattr(_sre_c, "POSSESSIVE_REPEAT", _sre_c.MAX_REPEAT),
    ):
        mn, mx, sub = av
        body_lo, body_hi = _seq_widths(sub.data, gw)
        if body_hi != 1:
            # A multi-char body RETREATS to an iteration boundary when cut and
            # ends short of the edge (round 1, finding 1) — no alarm.
            return False
        if init_hi > thresh:
            # The prefix alone may not fit the worst runway (byte-inflated
            # prefixes are exactly round 2's counterexample b).
            return False
        if _sat(init_hi + int(mn)) > thresh:
            # The engaged minimum must fit, or the attempt fails traceless
            # (round 1 finding 2, generalised).
            return False
        return True
    return False  # the wide part is not carried by a tail repeat: out


def _tier2_span_unprovable(pattern: str, flags: int, overlap_bytes: int) -> bool:
    """True when tier 2 cannot PROVE this pattern exact on a windowed subject.

    Routing on True costs one tier-3 materialisation and stays exact; a wrong
    False is the silent-wrong-answer class this file fears, so every branch
    errs toward True.
    """
    thresh = max(1, overlap_bytes // 4)  # chars; a codepoint is <= 4 bytes
    try:
        parsed = _sre_p.parse(pattern, flags)
    except Exception:  # noqa: BLE001 - unparseable here -> let tier 3 decide
        return True
    gw: dict = {}
    _seq_widths(parsed.data, gw)  # populate group widths for backrefs
    return not _admissible(parsed.data, 0, thresh, gw)


def _at_char_boundary(data, byte_off: int, nbytes: int) -> bool:
    """A byte offset is a CHARACTER position only if it is not a continuation
    byte. End-of-buffer always is one."""
    if byte_off >= nbytes:
        return True
    return (data[byte_off] & 0xC0) != 0x80


class Untranslatable(Exception):
    """The pattern has a construct tier 1 will not claim to translate exactly."""


def translate_pattern(pattern: str, flags: int = 0) -> bytes | None:
    """Exact bytes equivalent of a str pattern, or ``None`` if not claimable.

    Returning ``None`` is never a failure — it routes to tier 2, which is exact
    by construction. The translator therefore only ever has to be *sound*, and
    every construct it does not recognise is refused rather than guessed.
    """
    if flags & ~_KNOWN_FLAGS:
        return None  # a flag whose bytes-mode meaning has not been reasoned about
    if flags & re.VERBOSE:
        return None
    ignorecase = bool(flags & re.IGNORECASE)
    dotall = bool(flags & re.DOTALL)
    # `re.A` does not only narrow `\w \d \s \b` — it narrows IGNORECASE too:
    # `re.fullmatch("k", "\u212a", re.I)` matches, and with `+re.A` it does not.
    # Emitting the fold partners under re.A is a SPURIOUS match, so the round-1
    # note that "ASCII only narrows what this translator already refuses" was
    # wrong on its own terms.
    fold_unicode = not (flags & re.ASCII)
    # A quantifier anywhere after an unescaped `.` can reach the expansion
    # through a group — `(?:.){0,200}` slipped past a one-character lookahead
    # and measured 32.5x the original pattern. Deliberately over-broad: an
    # unnecessary refusal costs a tier-2 window, a missed one costs a sweep.
    if _has_quantified_dot(pattern):
        return None
    # MULTILINE has to be CARRIED, not just tolerated. The translated pattern is
    # compiled by the caller with no flags, so a dropped `re.M` silently changes
    # what `^`/`$` mean: `re.finditer(r"^a", ctx, re.M)` over "b\na\nc\na" then
    # returns [] where CPython returns two matches. Embedding it in the pattern
    # keeps the result self-contained — whoever compiles it gets the semantics.
    # `\n` is 0x0A and cannot occur inside a multi-byte UTF-8 sequence, so the
    # bytes-mode meaning is identical.
    prefix = b"(?m)" if flags & re.MULTILINE else b""
    out: list[bytes] = []
    i = 0
    n = len(pattern)
    try:
        while i < n:
            ch = pattern[i]
            if ch == "\\":
                if i + 1 >= n:
                    raise Untranslatable("trailing backslash")
                nxt = pattern[i + 1]
                if nxt.isdigit():
                    if ignorecase:
                        # In str mode a backreference re-matches case-
                        # insensitively under re.I; the copied bytes form would
                        # demand the exact bytes, so `(ab)\1` would miss "abAB".
                        raise Untranslatable("backreference under re.I")
                    out.append(pattern[i : i + 2].encode("ascii"))
                elif nxt in ("s", "d", "w", "S", "D", "W"):
                    # The Unicode classes ARE translatable (their non-ASCII
                    # members are enumerable), but only as large alternations,
                    # which is the pathology above. Tier 2 runs the original
                    # pattern on a bounded window and is exact anyway, so the
                    # right move is to refuse rather than to be exact and slow.
                    raise Untranslatable(f"\\{nxt}")
                elif nxt in _CLASS_MAP and _CLASS_MAP[nxt][0] is not None:
                    out.append(_CLASS_MAP[nxt][0])
                elif not nxt.isalnum():
                    # `\<char>` is still a literal, so it must take the SAME
                    # ignorecase treatment — including the non-ASCII refusal.
                    # Passing False here let `\\u24b6` reach tier 1 under re.I
                    # and miss `\\u24d0`.
                    out.append(_lit_bytes(nxt, ignorecase, fold_unicode))
                else:
                    raise Untranslatable(f"\\{nxt}")
                i += 2
                continue
            if ch == ".":
                # A quantified `.` does not belong on tier 1 at all. One
                # codepoint is two tokens however it is grouped, and the engine
                # must try every repeat count at every start position over the
                # WHOLE mapping: measured at a flat ~30x the original str
                # pattern across bounds 8…512, and `.{0,1000}X.{0,2000}` took
                # 19 s on 1 MiB — hours on a 31 MiB context.
                #
                # Tier 2 runs the ORIGINAL pattern on a bounded window and is
                # exact, so refusing costs a window and buys back the 30x. This
                # is also what round 1 did by accident: the ungrouped expansion
                # could not compile under a quantifier, so these patterns always
                # fell through to tier 2 — which is why its ladder never saw the
                # cost that the corrected binding exposed.
                if i + 1 < n and pattern[i + 1] in "*+?{":
                    raise Untranslatable("quantified `.`")
                out.append(_UTF8_CHAR if dotall else _UTF8_CHAR_NO_NL)
                i += 1
                continue
            if ch == "[":
                j = i + 1
                if j < n and pattern[j] == "^":
                    # A negated class in bytes mode matches one BYTE, so it can
                    # select a continuation byte. Expressible, not claimed.
                    raise Untranslatable("negated class")
                if j < n and pattern[j] == "]":
                    j += 1
                while j < n and pattern[j] != "]":
                    if pattern[j] == "\\":
                        j += 1
                    j += 1
                if j >= n:
                    raise Untranslatable("unterminated class")
                body = pattern[i + 1 : j]
                if not body.isascii() or "\\" in body:
                    raise Untranslatable("non-ascii or escaped class member")
                if ignorecase and fold_unicode and any(
                    re.fullmatch(f"[{body}]", c, re.IGNORECASE) for c in "isk"
                ):
                    # An i/s/k member under re.I would also match its non-ASCII
                    # fold partner in str mode. Correct expansion exists; tier 2
                    # is exact and simpler, so refuse rather than get it subtly
                    # wrong (the silent-divergence class this design fears).
                    raise Untranslatable("ignorecase class with an i/s/k member")
                cls = b"[" + body.encode("ascii") + b"]"
                # The flag was read for literals and silently dropped here, so
                # `[0-9a-f]` under re.I was emitted case-SENSITIVE and missed
                # "A-F". Scoping the fold to the class keeps it exact: bytes
                # mode folds ASCII only, and the i/s/k guard above is what makes
                # ASCII-only folding the same answer str mode would give.
                out.append(b"(?i:" + cls + b")" if ignorecase else cls)
                i = j + 1
                continue
            if ch == "(":
                # Lookbehind needs a FIXED width and a UTF-8 codepoint is
                # variable-width, so it cannot be translated at all.
                for opener in ("(?<=", "(?<!"):
                    if pattern.startswith(opener, i):
                        raise Untranslatable("lookbehind")
                matched = None
                for opener in ("(?:", "(?=", "(?!", "(?P="):
                    if pattern.startswith(opener, i):
                        matched = opener
                        break
                if matched is None and pattern.startswith("(?P<", i):
                    close = pattern.find(">", i)
                    if close == -1:
                        raise Untranslatable("unterminated group name")
                    matched = pattern[i : close + 1]
                if matched is None:
                    if pattern.startswith("(?", i):
                        raise Untranslatable("unsupported group flavour")
                    matched = "("
                if not matched.isascii():
                    raise Untranslatable("group prefix")
                out.append(matched.encode("ascii"))
                i += len(matched)
                continue
            if ch in ")|*+?^$":
                out.append(ch.encode("ascii"))
                i += 1
                continue
            if ch == "{":
                j = pattern.find("}", i)
                if j == -1:
                    raise Untranslatable("unterminated {")
                out.append(pattern[i : j + 1].encode("ascii"))
                i = j + 1
                continue
            out.append(_lit_bytes(ch, ignorecase, fold_unicode))
            i += 1
    except Untranslatable:
        return None
    except Exception:  # noqa: BLE001 - a translator bug must degrade, not raise
        return None

    candidate = prefix + b"".join(out)
    try:
        re.compile(candidate)
    except re.error:
        return None
    return candidate


# --------------------------------------------------------------------------- #
# match proxy — byte spans in, CHAR spans out
# --------------------------------------------------------------------------- #
class ViewMatch:
    """What the frozen code receives from ``re.finditer(pat, ctx)``.

    The recorded trajectories consume matches almost entirely through offsets
    (WO009 E2: ``start`` 407 + ``end`` 65 uses against ``group`` 95), so the
    cheap path is the common one: ``start``/``end``/``span`` convert byte
    offsets to char offsets through the index and allocate nothing, while
    ``group()`` decodes only the matched range.

    ``.string`` is deliberately NOT served — it is the whole subject, so it is
    an escape, and it is logged as one.
    """

    __slots__ = ("_m", "_view", "_cs", "_ce")

    def __init__(self, view: TextView, m: re.Match, char_span=None):
        self._m = m
        self._view = view
        if char_span is None:
            self._cs = view._coords.to_native(m.start())
            self._ce = view._coords.to_native(m.end())
        else:
            self._cs, self._ce = char_span

    def start(self, group: int = 0) -> int:
        if group:
            return self._view._coords.to_native(self._m.start(group))
        return self._cs

    def end(self, group: int = 0) -> int:
        if group:
            return self._view._coords.to_native(self._m.end(group))
        return self._ce

    def span(self, group: int = 0) -> tuple[int, int]:
        return (self.start(group), self.end(group))

    def group(self, *args):
        raw = self._m.group(*args)
        if isinstance(raw, tuple):
            return tuple(_as_text(x) for x in raw)
        return _as_text(raw)

    def groups(self, default=None):
        return tuple(
            _as_text(g) if g is not None else default for g in self._m.groups()
        )

    def groupdict(self, default=None):
        return {
            k: (_as_text(v) if v is not None else default)
            for k, v in self._m.groupdict().items()
        }

    def __getitem__(self, key):
        return self.group(key)

    @property
    def re(self):
        return self._m.re

    @property
    def string(self):
        _log_escape("match.string")
        return self._view._materialize()

    def __repr__(self) -> str:
        return f"<ViewMatch span=({self._cs}, {self._ce})>"


def _as_text(value):
    """Decode a byte span back to text, REFUSING a span that split a character.

    Tier 1 only emits patterns that cannot match a partial codepoint, so this
    should be unreachable — which is exactly why it must not be `errors=
    "replace"`. That silently substitutes U+FFFD, turning a translator bug into
    a plausible-looking wrong answer in the caller's hands. A strict decode
    turns the same bug into an exception at the point it happens. (The published
    round-1 table claimed a continuation-byte guard here; there was none.)
    """
    if isinstance(value, (bytes, bytearray, memoryview)):
        raw = bytes(value)
        try:
            return raw.decode("utf-8")
        except UnicodeDecodeError as exc:  # pragma: no cover - guards a bug
            _log_escape("bug:match-split-a-character")
            raise ValueError(
                "a tier-1 byte match ended inside a UTF-8 character; the "
                "translated pattern is not character-atomic"
            ) from exc
    return value


# --------------------------------------------------------------------------- #
# L3 — the text view
# --------------------------------------------------------------------------- #
class TextView:
    """A lazy ``str``-alike over a mapped UTF-8 file.

    Every value it returns is a real ``str``, so downstream library code runs
    natively on materialised slices. What it refuses to do is materialise the
    *whole* context, and the three-tier search is what makes that possible for
    the one consumer WO003 proved is universal: C-level ``re``.
    """

    # Tier-2 window. 4 MiB of UTF-8 decodes to at most ~16 MiB of UCS-4 — an
    # order below the 125 MiB the whole context costs, and transient.
    WINDOW_BYTES = 4 << 20
    # 256 KiB — completion room for boundary-crossing matches. NOT larger than
    # every possible match (this corpus's largest document is 27x it): patterns
    # that could out-span it stay on tier 2 only when truncation is detectable
    # (tail position); the rest take tier 3 (`_tier2_span_unprovable`).
    OVERLAP_BYTES = 1 << 18

    def __init__(self, mapped: MappedBytes, coords: Coords, label: str = "context"):
        self._bytes = mapped
        self._coords = coords
        self._label = label
        self._materialized: str | None = None

    # -- construction ------------------------------------------------------
    @classmethod
    def open(cls, path: str | os.PathLike[str]) -> TextView:
        mapped = MappedBytes(path)
        return cls(mapped, Utf8Coords(memoryview(mapped.buffer)), Path(path).name)

    # -- escape hatch ------------------------------------------------------
    def _materialize(self) -> str:
        if self._materialized is None:
            self._materialized = self._decode(0, len(self))
        return self._materialized

    def _decode(self, start: int, end: int) -> str:
        b0 = self._coords.to_byte(start)
        b1 = self._coords.to_byte(end)
        return self._bytes.slice_bytes(b0, b1).decode("utf-8")

    # -- str surface (the E1c census set) ----------------------------------
    def __len__(self) -> int:
        return self._coords.length()

    def __getitem__(self, item):
        n = len(self)
        if isinstance(item, slice):
            start, stop, step = item.indices(n)
            if step == 1:
                return self._decode(start, max(start, stop))
            # A strided slice cannot be served from a byte range — it needs the
            # decoded whole. Round 1 called `_decode(0, n)` directly, so the
            # single most expensive thing the view can do materialised ~125 MiB
            # while logging NO escape and caching nothing: the census could not
            # see it, and a second strided slice paid the cost again.
            _log_escape("slice:step")
            return self._materialize()[item]
        idx = item + n if item < 0 else item
        if not 0 <= idx < n:
            raise IndexError("string index out of range")
        return self._decode(idx, idx + 1)

    def _adjust(self, start, end) -> tuple[int, int]:
        """CPython's ADJUST_INDICES, replicated exactly.

        The asymmetry is the whole point and it is not a typo: ``end`` is
        clamped down to ``len`` but ``start`` is **not**. That is what makes
        ``"abc".find("", 99)`` return -1 rather than 3 — a difference a naive
        implementation gets wrong silently, and one WO003's review caught the
        hard way (its first fix covered only inverted ranges and left 324
        mismatches behind).
        """
        n = len(self)
        if end is None:
            e = n
        elif end > n:
            e = n
        elif end < 0:
            e = max(0, end + n)
        else:
            e = end
        if start is None:
            s = 0
        elif start < 0:
            s = max(0, start + n)
        else:
            s = start
        return s, e

    def _find(self, sub: str, start, end, *, last: bool) -> int:
        s, e = self._adjust(start, end)
        if s > e:
            return -1
        if not sub:
            return e if last else s
        needle = sub.encode("utf-8")
        b0, b1 = self._coords.to_byte(s), self._coords.to_byte(e)
        # mmap.find/rfind search the mapping IN PLACE. Slicing it first would
        # copy the whole range into a bytes object on every call — 31 MiB per
        # find on this corpus, which is both the anon cost this view exists to
        # avoid and, at thousands of calls per session, the dominant runtime.
        buf = self._bytes.buffer
        hit = buf.rfind(needle, b0, b1) if last else buf.find(needle, b0, b1)
        if hit < 0:
            return -1
        return self._coords.to_native(hit)

    def find(self, sub, start=None, end=None) -> int:
        return self._find(str(sub), start, end, last=False)

    def rfind(self, sub, start=None, end=None) -> int:
        return self._find(str(sub), start, end, last=True)

    def index(self, sub, start=None, end=None) -> int:
        out = self.find(sub, start, end)
        if out < 0:
            raise ValueError("substring not found")
        return out

    def rindex(self, sub, start=None, end=None) -> int:
        out = self.rfind(sub, start, end)
        if out < 0:
            raise ValueError("substring not found")
        return out

    def count(self, sub, start=None, end=None) -> int:
        s, e = self._adjust(start, end)
        if s > e:
            return 0
        sub = str(sub)
        if not sub:
            # The empty needle sits between every pair of CHARACTERS, so this
            # has to be counted in char space; counting it in bytes over-reports
            # by exactly the number of continuation bytes in the range.
            return e - s + 1
        b0, b1 = self._coords.to_byte(s), self._coords.to_byte(e)
        needle = sub.encode("utf-8")
        buf = self._bytes.buffer
        total = 0
        at = buf.find(needle, b0, b1)
        while at >= 0:
            total += 1
            at = buf.find(needle, at + len(needle), b1)
        return total

    def __contains__(self, sub) -> bool:
        return self.find(str(sub)) >= 0

    def __eq__(self, other) -> bool:
        if other is self:
            return True
        # Compared in chunks: a whole-mapping copy here would be the exact cost
        # the view exists to avoid, and equality is answered by the first
        # differing chunk anyway.
        if isinstance(other, TextView):
            if len(self) != len(other) or len(self._bytes) != len(other._bytes):
                return False
            step = 1 << 20
            for at in range(0, len(self._bytes), step):
                if self._bytes.slice_bytes(at, at + step) != other._bytes.slice_bytes(
                    at, at + step
                ):
                    return False
            return True
        if isinstance(other, str):
            if len(other) != len(self):
                return False
            encoded = other.encode("utf-8")
            if len(encoded) != len(self._bytes):
                return False
            step = 1 << 20
            for at in range(0, len(encoded), step):
                if self._bytes.slice_bytes(at, at + step) != encoded[at : at + step]:
                    return False
            return True
        return NotImplemented

    def __hash__(self) -> int:
        # `__eq__` answers True against an equal `str`, so hashing on a 4 KiB
        # prefix broke the contract: `d[some_str] = 1; d[view]` raised KeyError
        # for objects that compare equal. Hashing IS a whole-object operation —
        # priced and logged like `__str__`, not faked cheaply.
        _log_escape("__hash__")
        return hash(self._materialize())

    def __bool__(self) -> bool:
        return len(self) > 0

    def __str__(self) -> str:
        _log_escape("__str__")
        return self._materialize()

    def __repr__(self) -> str:
        return f"<TextView {self._label!r} chars={len(self)}>"

    def __getattr__(self, name: str):
        # Anything not served natively falls back to the real str, logged as the
        # façade cost FR002 requires. Never a silent wrong answer.
        if name.startswith("_"):
            raise AttributeError(name)
        if not hasattr(str, name):
            # A `hasattr(ctx, "read")` duck-type PROBE is a question, not a use.
            # Round 1 answered it by materialising ~125 MiB and then raising
            # AttributeError anyway — the most expensive possible way to say no.
            raise AttributeError(name)
        _log_escape(f"attr:{name}")
        return getattr(self._materialize(), name)

    @property
    def context_char_count(self) -> int:
        """Duck-type hook ``QueryMetadata`` reads. It is the TRUE char count.

        WO003's PathRef control passed ``st_size`` — BYTES — here, and that
        number is interpolated verbatim into the root-LM system prompt
        ("Your context is a str of N total characters", rlm/utils/prompts.py).
        On this corpus bytes exceed chars by 0.19-0.27%, so that lane's prompt
        differed from the recorded run's. Replay could not see it:
        ``ReplaySession.next()`` is an ordered cursor that discards the prompt.
        """
        return len(self)

    # -- the re protocol (three tiers) -------------------------------------
    def _finditer(self, pattern: str, flags: int, api: str):
        translated = translate_pattern(pattern, flags)
        if translated is not None:
            _log_tier("tier1", api)
            data = self._bytes
            n = len(data)
            for m in re.finditer(translated, data.buffer):
                # Every guard before this one constrained matched CONTENT. A
                # zero-width match constrains nothing, so `[abc]*` matched the
                # empty string at each of the three continuation bytes inside
                # `\U0001d54f` — three spurious matches at one character
                # position, on the +0.0 MiB path, invisible to `_as_text`
                # because b"" decodes cleanly. A byte offset is a character
                # position only when it is not a continuation byte.
                if not (
                    _at_char_boundary(data, m.start(), n)
                    and _at_char_boundary(data, m.end(), n)
                ):
                    continue
                yield ViewMatch(self, m)
            return
        _log_tier("tier2", api)
        yield from self._windowed_finditer(pattern, flags)

    def _windowed_finditer(self, pattern: str, flags: int):
        """The ORIGINAL pattern and the ORIGINAL engine on a decoded window.

        **The precondition, now ENFORCED by partition rather than stated.**
        Tier 2 is exact only while no match exceeds ``OVERLAP_BYTES`` (256 KiB
        as configured; window growth briefly raised this to 2 x WINDOW_BYTES
        and was **removed in round 10** — it cost one task 1,097 s against
        281 s and tripped G3). Historically nothing verified it, and a
        violation was a **silent miss or a wrong span, with no log line**. The
        violation cannot be inferred from output: the event is a match that
        FAILED to complete, absent from the window's output and from any
        larger probe's, so every proxy tried across nine review rounds read
        "nothing here" and "nothing here because it broke" identically.

        **This corpus VIOLATES the unpartitioned tier's precondition.**
        Measured over the 46 recorded prompts (46,000 ``<document ...>``
        blocks): mean 33,212 characters, median 10,329 — but **maximum
        7,094,565** (``951.txt``), which is **27x the overlap**, and **46 of
        46** files hold a document larger than it. On the unpartitioned tier,
        tasks **1203 and 239** ran document-spanning patterns here and found
        **997 and 998 of 1,000** — silently. An earlier version of this note
        claimed a "~32 KB longest match, three orders below the bound" — that
        was the **mean**; a precondition needs the **maximum**, so the margin
        was **0.04x**, not 8x. G3 cannot see any of this (the oracle compares
        the replayed call sequence and the final answer, not search results),
        and it is **not** a consequence of removing growth — the ladder-
        measured code had none either.

        **The partition (the successor item named by FR006-CL004, built
        2026-08-03; admission rule made COMPOSITE by ABR 260803-2100 rounds
        1-2).** ``_tier2_span_unprovable`` admits a pattern to this tier only
        when its exactness is provable against the window geometry:

        * **composite-narrow** — the SUMMED worst-case width fits the overlap
          (chars vs ``overlap // 4``, so the ×4 UTF-8 bound covers the sum in
          bytes); or
        * **narrow prefix + one width-1-body tail repeat** — everything before
          the final repeat fits the overlap, the prefix plus the repeat's
          engaged MINIMUM fits, and the body consumes exactly one character
          per iteration, so a truncated match necessarily ends at the
          window's last character — the ``m.end() == len(text)`` signal
          below — and escalates to the whole subject.

        Everything else — wide non-tail (the 1203/239 shapes), multi-char
        iteration bodies (they retreat past the edge signal), wide minimums
        however wrapped, and compositions of narrow pieces whose sum is wide —
        takes tier 3, logged ``window:span-unbounded``, exact by
        construction. The 2g density ladder was measured BEFORE this
        partition existed; its rows describe the unpartitioned tier and
        re-measurement is the successor WO's business.
        """
        if _needs_full_subject(pattern):
            _log_escape("window:needs-full-subject")
            for m in re.finditer(pattern, self._materialize(), flags):
                yield ViewMatch(self, m, char_span=(m.start(), m.end()))
            return

        if _tier2_span_unprovable(pattern, flags, self.OVERLAP_BYTES):
            # The pattern could out-span the overlap and its truncation would
            # be unobservable — the whole subject is the only sound answer.
            _log_escape("window:span-unbounded")
            for m in re.finditer(pattern, self._materialize(), flags):
                yield ViewMatch(self, m, char_span=(m.start(), m.end()))
            return

        rx = re.compile(pattern, flags)
        n_bytes = len(self._bytes)
        if n_bytes == 0:
            # `while b_start < n_bytes` never runs, so an empty subject returned
            # nothing where CPython yields the one legal empty match.
            for m in rx.finditer(""):
                yield ViewMatch(self, m, char_span=(m.start(), m.end()))
            return
        b_start = 0
        owned_start = 0
        # The high-water mark of what has already been YIELDED. Ownership alone
        # is not enough: a match that starts before the boundary and runs past
        # it is emitted by the window that owns its start, and the next window —
        # whose subject begins at that boundary — then finds the tail as a match
        # of its own. It is a fragment, not a duplicate, so a span set cannot
        # absorb it. A position already inside an emitted match can never start
        # another one, which is what this enforces.
        covered_upto = 0
        emitted: set[tuple[int, int]] = set()
        while b_start < n_bytes:
            b_start = self._align(b_start)
            b_end = self._align(min(b_start + self.WINDOW_BYTES, n_bytes))
            text = self._bytes.slice_bytes(b_start, b_end).decode("utf-8")
            # A match that FAILS to complete inside the window leaves no trace,
            # so no property of what was emitted can detect it — round 5 and
            # round 6 both tried and both were separable by counterexample. The
            # only sound test is a property of the WINDOW: extend it and see
            # whether the owned region's answer changes. If two successive
            # window sizes agree, the region is stable; if they disagree, the
            # larger one is right and we keep going. This terminates at the file
            # end, where it degenerates to tier 3 — exact, and only for the
            # pattern that actually needs it.
            # The overlap must be able to hold a whole match; when it cannot,
            # windowing has no guarantee to offer and the honest answer is the
            # whole subject. Six rounds of review have shown that every attempt
            # to INFER this from what a scan produced is separable by
            # counterexample, so it is enforced as a precondition instead: the
            # ownership boundary never sits closer to the window's end than the
            # overlap, and a window that still truncates after growing to the
            # file is tier 3 by construction.
            last = b_end >= n_bytes
            base_char = self._coords.to_native(b_start)
            owned_end = (
                base_char + len(text)
                if last
                else self._coords.to_native(
                    self._align(max(b_start + 1, b_end - self.OVERLAP_BYTES))
                )
            )
            # The scan must START at the owned position, not be filtered after
            # the fact. A greedy match found earlier and discarded is not the
            # same as the match a correctly-positioned scan produces: with
            # `\w+\s+\w+` the two differ, which a filter cannot repair.
            floor = max(owned_start, covered_upto)
            scan_from = max(0, floor - base_char)
            if scan_from >= len(text):
                if last:
                    break
                owned_start = max(owned_start, owned_end)
                b_start = b_end - self.OVERLAP_BYTES
                continue
            for m in rx.finditer(text, scan_from):
                cs, ce = base_char + m.start(), base_char + m.end()
                # The final position of the subject is a legal match start
                # (`re.finditer(r"\w*", "abc")` ends with `(3, 3)`), and an
                # exclusive bound on the LAST window silently dropped it — at
                # every window size, including a single-window file.
                if cs >= owned_end + (1 if last else 0):
                    continue
                if m.end() == len(text) and not last:
                    # Possibly truncated by the window edge — fall through to a
                    # full-materialise pass rather than report a short match.
                    _log_escape("window:match-spans-edge")
                    for full in re.finditer(pattern, self._materialize(), flags):
                        key = (full.start(), full.end())
                        if key not in emitted:
                            emitted.add(key)
                            yield ViewMatch(self, full, char_span=key)
                    return
                emitted.add((cs, ce))
                covered_upto = max(covered_upto, ce)
                yield ViewMatch(self, m, char_span=(cs, ce))
            del text
            if last:
                break
            # Advance only to what was actually COVERED, never to `owned_end`.
            # The overlap rescues a match that can be *completed* inside the
            # window; it does nothing for one the truncation makes *fail*, which
            # emits no match and no signal. Advancing to `owned_end` then let the
            # next window start past that match's true beginning — it reported
            # (320, 340) where CPython gives (319, 340). Advancing to
            # `covered_upto` instead lets the next window, which begins earlier
            # and therefore carries left context, find it whole. Re-emission is
            # already impossible: a position inside an emitted match cannot start
            # another one.
            owned_start = max(owned_start, covered_upto)
            # The overlap is only sufficient if it reaches back to everything we
            # could not prove clean. Round 4 showed exactness resting on an
            # UNCHECKED precondition ("overlap > longest match"), true at the
            # production 4 MiB / 256 KiB geometry and false at 64 B / 16 B — so
            # check it instead of asserting it in a comment. When it fails the
            # answer is still exact, just slower: tier 3 is the whole subject.
            # The precondition is "no match is longer than the overlap", and the
            # round-4 fix checked a proxy that is true in the ORDINARY case —
            # the last match ending before the boundary — so tier 2 collapsed
            # into materialisation for 5 of 9 realistic patterns at production
            # geometry. A match can only be lost across a boundary if it spans
            # more than the overlap, so that is what to watch: cheap, and it
            # fires on the pathology instead of on the common path.
            b_start = b_end - self.OVERLAP_BYTES

    def _align(self, byte_off: int) -> int:
        """Nudge to a codepoint boundary so a window never splits a character."""
        n = len(self._bytes)
        if byte_off <= 0:
            return 0
        if byte_off >= n:
            return n
        while byte_off < n and (self._bytes[byte_off] & 0xC0) == 0x80:
            byte_off += 1
        return byte_off

    # public re-protocol entry points, used by the import shim
    def _re_finditer(self, pattern, flags=0):
        return self._finditer(pattern, flags, "finditer")

    def _re_search(self, pattern, flags=0):
        for m in self._finditer(pattern, flags, "search"):
            return m
        return None

    def _re_findall(self, pattern, flags=0):
        out = []
        for m in self._finditer(pattern, flags, "findall"):
            groups = m.groups()
            # CPython yields '' for a group that did not participate, never
            # None: `re.findall(r"x(y)?", "x xy")` is `['', 'y']`.
            groups = tuple("" if g is None else g for g in groups)
            out.append(
                m.group(0) if not groups else (groups[0] if len(groups) == 1 else groups)
            )
        return out

    def _re_split(self, pattern, flags=0, maxsplit=0):
        _log_escape("c-level:re.split")
        return re.split(pattern, self._materialize(), maxsplit=maxsplit, flags=flags)


# --------------------------------------------------------------------------- #
# handles
# --------------------------------------------------------------------------- #
def utf8_char_count(path: str | os.PathLike[str], chunk_size: int = 4 << 20) -> int:
    """Codepoint count of a UTF-8 file, streamed — peak cost is one chunk.

    Exists because the count is interpolated verbatim into the root-LM system
    prompt, so it has to be the true char count and cannot be approximated by
    the file size (see ``TextView.context_char_count``).
    """
    total = 0
    with open(path, "rb") as fh:
        while True:
            block = fh.read(chunk_size)
            if not block:
                break
            total += _count_chars(block)
    return total


class PathRef:
    """Path handle — WO003's cheap-alternative control, kept as the transport.

    ``chars`` must be the TRUE codepoint count (see
    ``TextView.context_char_count``).
    """

    __slots__ = ("path", "chars")

    def __init__(self, path: str, chars: int):
        self.path = path
        self.chars = chars

    @property
    def context_char_count(self) -> int:
        return self.chars

    def __repr__(self) -> str:
        return f"PathRef({self.path!r}, {self.chars})"


class ViewRef(PathRef):
    """Same transport, but the guest binds a :class:`TextView` instead of
    running ``f.read()``. This is the WO009 treatment lane."""

    def open(self) -> TextView:
        return TextView.open(self.path)


# --------------------------------------------------------------------------- #
# the interception point (kept from WO003; its ACTION is what changed)
# --------------------------------------------------------------------------- #
# WO003 proved this shim is necessary and sufficient to reach C-level `re`:
# the frozen code calls `re.finditer(<str pattern>, ctx)`, and a str pattern
# over a buffer object raises no matter what the object is, so the buffer path
# is reachable ONLY through interception. What changed in WO009 is that the
# shim now SERVES the call from the mapping instead of materialising for it.
_TEXT_MODULES = frozenset(
    {"re", "json", "difflib", "textwrap", "csv", "string", "html", "unicodedata"}
)
_RE_DISPATCH = {
    "finditer": "_re_finditer",
    "search": "_re_search",
    "findall": "_re_findall",
    "split": "_re_split",
}

# Every other `re` entry point that takes a subject. These keep WO003's shim
# behaviour — materialise and LOG — because the alternative is what round 1
# shipped: the raw C function receives a TextView and raises TypeError, so
# `re.match(p, ctx)` crashed the guest where WO003 merely escaped. `sub`/`subn`
# belong here rather than in the lazy table because their subject is the THIRD
# positional argument, which the two-positional binding above never inspected.
_RE_COERCE = frozenset({"match", "fullmatch", "sub", "subn"})


def _coerce(value, where: str):
    if isinstance(value, TextView):
        _log_escape(where)
        return value._materialize()
    return value


def _coercing_call(target, where: str):
    """Run a C-level callable that cannot take a view: materialise, then log.

    The subject's position differs per function (`re.sub` puts it third), so
    every argument is inspected rather than one being assumed.
    """

    def call(*args, **kwargs):
        args = tuple(
            _coerce(a, where) if isinstance(a, TextView) else a for a in args
        )
        kwargs = {
            k: (_coerce(v, where) if isinstance(v, TextView) else v)
            for k, v in kwargs.items()
        }
        return target(*args, **kwargs)

    return call


class _PatternProxy:
    """``p = re.compile(pat); p.finditer(ctx)`` has to reach the same tiers."""

    __slots__ = ("_p",)

    def __init__(self, pattern):
        self._p = pattern

    def __getattr__(self, name):
        target = getattr(self._p, name)
        if name in _RE_COERCE:
            return _coercing_call(target, f"re.Pattern.{name}")
        if name not in _RE_DISPATCH:
            return target

        def call(subject, *args, **kwargs):
            if isinstance(subject, TextView):
                method = getattr(subject, _RE_DISPATCH[name])
                return method(self._p.pattern, self._p.flags, *args, **kwargs)
            return target(subject, *args, **kwargs)

        return call


class _ReProxy:
    __slots__ = ("_m",)

    def __init__(self, module):
        self._m = module

    def __getattr__(self, name):
        target = getattr(self._m, name)
        if name == "compile":

            def compile_(pattern, flags=0):
                return _PatternProxy(target(pattern, flags))

            return compile_
        if name in _RE_DISPATCH:

            def call(pattern, subject, *args, **kwargs):
                if isinstance(subject, TextView):
                    method = getattr(subject, _RE_DISPATCH[name])
                    if name == "split":
                        # re.split(pattern, string, maxsplit=0, flags=0) — the
                        # positional order is NOT the view method's.
                        maxsplit = kwargs.pop("maxsplit", args[0] if args else 0)
                        flags = kwargs.pop("flags", args[1] if len(args) > 1 else 0)
                        return method(pattern, flags, maxsplit)
                    flags = kwargs.pop("flags", args[0] if args else 0)
                    return method(pattern, flags)
                return target(pattern, subject, *args, **kwargs)

            return call
        if name in _RE_COERCE:
            return _coercing_call(target, f"re.{name}")
        return target


class _CoercingProxy:
    """Other C-level text modules keep WO003's behaviour: materialise + log."""

    __slots__ = ("_m", "_name")

    def __init__(self, module, name: str):
        self._m = module
        self._name = name

    def __getattr__(self, name):
        attr = getattr(self._m, name)
        # Classes pass through UNWRAPPED, deliberately. Wrapping them coerces
        # constructor args, but a proxy is not a class: `except
        # json.JSONDecodeError` becomes an uncatchable TypeError and
        # `isinstance(node, ast.Assign)` silently flips True->False
        # (BR002-WO003 ABR R2-1, reverted there for exactly this reason).
        if callable(attr) and not isinstance(attr, type):
            where = f"c-level:{self._name}.{name}"

            def call(*args, **kwargs):
                args = tuple(_coerce(a, where) for a in args)
                kwargs = {k: _coerce(v, where) for k, v in kwargs.items()}
                return attr(*args, **kwargs)

            return call
        return attr


def make_coercing_import(real_import=None):
    """Guest ``__import__`` that routes text modules through the proxies."""
    import builtins

    base = real_import or builtins.__import__

    def _import(name, globals=None, locals=None, fromlist=(), level=0):
        module = base(name, globals, locals, fromlist, level)
        root = name.split(".")[0]
        if root not in _TEXT_MODULES:
            return module
        if root == "re":
            return _ReProxy(module)
        return _CoercingProxy(module, root)

    return _import
