"""Tests for the lazy context view (RLM Lab, BR002-WO009).

The equivalence sweep here is the load-bearing test of the whole design, not a
formality: a transparent layer is free for the caller and expensive for the
implementer, and when it is wrong it is wrong *silently*. WO003's adversarial
review found exactly that class twice — `"abc".find("", 99)` returning 3 instead
of -1, and two views over identical content comparing False — so every str-facing
operation is checked against real ``str`` over content that carries astral
emoji, combining marks and CJK, not over ASCII.
"""

from __future__ import annotations

import random
import re

import pytest

from rlm.context_view import (
    _FOLD_EXTRA,
    IdentityCoords,
    MappedBytes,
    TextView,
    Utf8Coords,
    ViewRef,
    escape_reset,
    escape_snapshot,
    make_coercing_import,
    search_tier_snapshot,
    translate_pattern,
    utf8_char_count,
)

# Deliberately nasty: astral emoji (4 bytes), CJK (3), accents (2), a combining
# mark, and an ASCII run long enough that block boundaries land inside it.
SAMPLE = (
    "the leukemia trial \U0001F92F and café data 임상시험 "
    "é combining \U0001FAD2 more ascii padding here 0123456789 "
) * 400


@pytest.fixture
def view(tmp_path):
    path = tmp_path / "ctx.txt"
    path.write_text(SAMPLE, encoding="utf-8")
    return TextView.open(path)


# --------------------------------------------------------------------------- #
# coordinates
# --------------------------------------------------------------------------- #
def test_utf8coords_roundtrips_every_char_boundary():
    data = SAMPLE.encode("utf-8")
    coords = Utf8Coords(memoryview(data), block=64)  # tiny block => many scans
    assert coords.length() == len(SAMPLE)
    for pos in range(0, len(SAMPLE), 97):
        want_byte = len(SAMPLE[:pos].encode("utf-8"))
        assert coords.to_byte(pos) == want_byte
        assert coords.to_native(want_byte) == pos


def test_utf8coords_clamps_out_of_range():
    data = "abc\U0001F92F".encode()
    coords = Utf8Coords(memoryview(data))
    assert coords.to_byte(-5) == 0
    assert coords.to_byte(10_000) == len(data)
    assert coords.to_native(-1) == 0
    assert coords.to_native(10_000) == 4


def test_identity_coords_is_the_generalisation_proof():
    """The L2 interface has to hold for a modality with no char notion at all —
    an image or an audio buffer needs this and nothing else."""
    coords = IdentityCoords(1234)
    assert coords.length() == 1234
    for pos in (0, 1, 999, 1234):
        assert coords.to_byte(pos) == pos
        assert coords.to_native(pos) == pos


# --------------------------------------------------------------------------- #
# str surface equivalence
# --------------------------------------------------------------------------- #
def test_len_and_char_count_are_the_true_codepoint_count(view, tmp_path):
    assert len(view) == len(SAMPLE)
    # NOT the byte size: WO003's control passed st_size here and that number is
    # interpolated into the root-LM system prompt.
    assert len(SAMPLE.encode("utf-8")) != len(SAMPLE)
    assert view.context_char_count == len(SAMPLE)
    assert utf8_char_count(tmp_path / "ctx.txt") == len(SAMPLE)


def test_slices_match_str_including_negative_and_open_ended(view):
    rng = random.Random(0)
    n = len(SAMPLE)
    cases = [(0, n), (0, 1), (n - 1, n), (n, n), (-50, -1), (-1, None), (None, 20)]
    cases += [
        tuple(sorted((rng.randrange(-n, n), rng.randrange(-n, n))))
        for _ in range(200)
    ]
    for start, stop in cases:
        assert view[start:stop] == SAMPLE[start:stop], (start, stop)


def test_index_access_matches_str(view):
    for i in (0, 1, 19, len(SAMPLE) - 1, -1, -20):
        assert view[i] == SAMPLE[i]
    with pytest.raises(IndexError):
        view[len(SAMPLE)]


@pytest.mark.parametrize("needle", ["leukemia", "café", "\U0001F92F", "zzz", ""])
def test_find_rfind_count_match_str_over_a_start_end_sweep(view, needle):
    n = len(SAMPLE)
    bounds = [None, 0, 1, 10, n // 2, n - 1, n, n + 99, -1, -n, -n - 5]
    for start in bounds:
        for end in bounds:
            assert view.find(needle, start, end) == SAMPLE.find(needle, start, end)
            assert view.rfind(needle, start, end) == SAMPLE.rfind(needle, start, end)
            assert view.count(needle, start, end) == SAMPLE.count(needle, start, end)


def test_empty_needle_start_is_not_clamped_down_to_len(view):
    """``"abc".find("", 99)`` is -1, not 3 — the WO003 regression, pinned."""
    assert view.find("", len(SAMPLE) + 99) == SAMPLE.find("", len(SAMPLE) + 99) == -1
    assert view.find("", len(SAMPLE)) == SAMPLE.find("", len(SAMPLE))


def test_contains_and_index_match_str(view):
    assert ("leukemia" in view) is True
    assert ("not-in-corpus" in view) is False
    assert view.index("leukemia") == SAMPLE.index("leukemia")
    with pytest.raises(ValueError):
        view.index("not-in-corpus")


def test_view_to_view_equality_is_content_based(tmp_path):
    """Two views over identical content must compare equal — Python's identity
    fallback would silently answer False (WO003 ABR round 1)."""
    a, b = tmp_path / "a.txt", tmp_path / "b.txt"
    a.write_text(SAMPLE, encoding="utf-8")
    b.write_text(SAMPLE, encoding="utf-8")
    va, vb = TextView.open(a), TextView.open(b)
    assert va == vb
    assert va == SAMPLE
    assert not (va == SAMPLE + "x")
    assert bool(va) is True


# --------------------------------------------------------------------------- #
# pattern translation (tier 1)
# --------------------------------------------------------------------------- #
TRANSLATION_CASES = [
    "leukemia", "café", "\U0001F92F", r"\d+", r"\d{2,4}", r"\s+", r"a\sb",
    r"a.c", r".{0,3}trial", r"(leukemia|trial)", r"(?:data|rows)", r"^the",
    r"data$", r"[a-z]+", r"[A-Za-z0-9]{3}", r"tri[ae]l", r"<doc id=(\d+)",
    r"id=(?P<id>\d+)", r"\bdata\b", r"a\.c", r"(?=leuk)leukemia",
    r"(?!zzz)trial", "class", "kelvin",
    # --- quantifier binding (WO009 round-2 blocker #2) ------------------------
    # `.` expands to TWO tokens, so a following quantifier used to bind to the
    # expansion's tail instead of the character: `a.?b` lost the zero-char
    # branch and `a.+b` became a POSSESSIVE repeat. Both compiled.
    r"a.?b", r"a.+b", r"a.*b", r".{2,3}x", r"a.{1,2}?b",
    # A multi-byte literal is two tokens for the same reason: `é+` must not
    # compile as one lead byte and a repeated continuation byte.
    r"é+", r"é{2}", r"é?x", r"임상+",
    # --- flags the translator has to carry, not drop (blocker #1) -------------
    r"^a", r"a$", r"^the", r"data$",
    # --- ignorecase reaching classes and backrefs (blocker #4) ----------------
    r"[0-9a-f]+", r"[A-F]{2}", r"(ab)\1", r"(?P<w>tri)(?P=w)",
    # --- boundaries: must be REFUSED, never copied through (blocker #3) -------
    r"\bdata\b", r"\Bata", r"\bcafé\b",
]
FLAG_SETS = [
    0,
    re.IGNORECASE,
    re.DOTALL,
    re.IGNORECASE | re.DOTALL,
    # MULTILINE was absent, which is why a dropped `re.M` shipped: no case in
    # this table could observe `^`/`$` changing meaning.
    re.MULTILINE,
    re.MULTILINE | re.IGNORECASE,
]
DIFFERENTIAL_TEXTS = [
    "the leukemia trial x9 and data",
    "café naïve résumé Ångström",
    "백혈병 임상시험 2024",
    "emoji \U0001F92F here \U0001FAD2 and \U0001F600 more",
    "claſs act temp 100K Kırmak",  # long s, kelvin, dotless i
    "a b c",  # NBSP, EM SPACE
    "digits 42 ٩٨ done",  # arabic-indic
    "line one\nline two\ttab",
    # A divergence is only observable if some text can EXPOSE it. These are the
    # gaps that let round 1's defects through a green suite:
    "naïdata and data and dataé",       # non-ASCII adjoining a word -> \b
    "HEX 0xDEADBEEF then 0xdeadbeef",   # mixed case -> classes under re.I
    "b\na\nc\na\nthe start\ndata",    # real line structure -> re.M
    "aéébb ééé aéb ab axxb",            # multi-byte literal under a quantifier
    "abAB abab ABAB",                   # backreferences under re.I
]


@pytest.mark.parametrize("pattern", TRANSLATION_CASES)
@pytest.mark.parametrize("flags", FLAG_SETS)
def test_translated_pattern_finds_exactly_the_same_spans(pattern, flags):
    translated = translate_pattern(pattern, flags)
    if translated is None:
        pytest.skip("refused by the translator -> tier 2 serves it exactly")
    for text in DIFFERENTIAL_TEXTS:
        data = text.encode("utf-8")
        want = [(m.start(), m.end(), m.group()) for m in re.finditer(pattern, text, flags)]
        got = []
        for m in re.finditer(translated, data):
            start = len(data[: m.start()].decode("utf-8"))
            end = len(data[: m.end()].decode("utf-8"))
            got.append((start, end, m.group().decode("utf-8")))
        assert got == want, (pattern, flags, text)


def test_translator_refuses_rather_than_guesses():
    """Refusal is not failure — it routes to tier 2, which is exact. What must
    never happen is a confident wrong translation."""
    for pattern in [r"(?<=abc)x", r"[^a]", r"\S+", r"\W", r"(?i:x)"]:
        assert translate_pattern(pattern, 0) is None, pattern


def test_edge_sensitive_predicate_over_the_construct_set():
    """`_needs_full_subject` certified over the SET of constructs whose meaning
    is defined against the whole subject, not over two examples."""
    from rlm.context_view import _needs_full_subject

    bs = chr(92)
    edge = ["^a", "a$", bs + "Aa", "a" + bs + "Z", bs + "ba", "a" + bs + "B",
            "(?=a)b", "(?!a)b", "(?<=a)b", "(?<!a)b"]
    for construct in edge:
        assert _needs_full_subject(construct), construct
    # ...and NOT when the same characters are escaped literals or class members
    benign = ["a" + bs + "^b", "a" + bs + "$b", "[$^]x", "(?:ab)c", "a.b"]
    for construct in benign:
        assert not _needs_full_subject(construct), construct


def test_char_boundary_predicate_over_every_utf8_lead_and_continuation():
    """`_at_char_boundary` certified over every byte value, not six samples."""
    from rlm.context_view import _at_char_boundary

    data = bytes(range(256))
    for b in range(256):
        expected = (b & 0xC0) != 0x80
        assert _at_char_boundary(data, b, 256) is expected, hex(b)
    assert _at_char_boundary(data, 256, 256) is True  # end of buffer


def test_word_boundaries_are_refused_not_copied():
    """`\b` is Unicode-aware in str mode and ASCII-only in bytes mode, so every
    UTF-8 continuation byte reads as a non-word character and a boundary appears
    inside any non-ASCII word. Round 1 copied it through: `\bcat\b` on
    "naïcat and cat" returned 2 matches against str's 1. It cannot be expressed
    (variable-width lookbehind), so the only sound answer is refusal."""
    for pattern in [r"\bdata\b", r"\Bata", r"\bcafé\b", r"a\b", r"\B"]:
        assert translate_pattern(pattern, 0) is None, pattern


def test_flags_are_carried_or_refused_never_dropped():
    """A translated pattern is compiled by the caller with NO flags, so any flag
    the translator neither encodes nor refuses is silently lost. `re.M` was
    lost: `^a` over "b\na\nc\na" returned [] against CPython's two matches."""
    carried = translate_pattern("^a", re.MULTILINE)
    assert carried is not None and carried.startswith(b"(?m)")
    assert [m.span() for m in re.finditer(carried, b"b\na\nc\na")] == [(2, 3), (6, 7)]
    # A flag whose bytes-mode meaning has not been reasoned about must refuse.
    assert translate_pattern("abc", re.VERBOSE) is None
    assert translate_pattern("abc", re.DEBUG) is None


def test_ignorecase_reaches_classes_and_refuses_backreferences():
    """re.I was applied to literals and silently dropped for classes, so
    `[0-9a-f]+` under re.I was emitted case-SENSITIVE and missed "A-F"."""
    translated = translate_pattern("[0-9a-f]+", re.IGNORECASE)
    assert translated is not None
    subject = "XYZ ABC 12ef"
    want = [m.group() for m in re.finditer("[0-9a-f]+", subject, re.IGNORECASE)]
    got = [m.group().decode() for m in re.finditer(translated, subject.encode())]
    assert got == want
    # A backreference under re.I re-matches case-insensitively in str mode; the
    # copied bytes form would demand the exact bytes, so refuse.
    assert translate_pattern(r"(ab)\1", re.IGNORECASE) is None


@pytest.mark.parametrize("pattern", ["a.?b", "a.+b", "a.*b", ".{2,3}x", ".{0,200}z"])
def test_quantified_dot_is_refused_rather_than_translated(pattern):
    """Correct AND affordable are separate bars, and this one fails the second.

    One codepoint is two tokens however it is grouped, so a quantified `.` makes
    the engine try every repeat count at every start position over the whole
    mapping — a flat ~30x the original str pattern across bounds 8…512, and
    `.{0,1000}X.{0,2000}` measured 19 s on 1 MiB. Tier 2 runs the original
    pattern on a window and is exact, so refusal is the cheap answer, not a
    concession.
    """
    assert translate_pattern(pattern, 0) is None, pattern


@pytest.mark.parametrize(
    "pattern,subject",
    [("é+", "aéébb"), ("é{2}", "aéébb"), ("é?x", "éx"), ("(?:ab)+c", "ababc")],
)
def test_quantifiers_bind_to_the_character_not_the_expansion(pattern, subject):
    """`.` and any multi-byte literal expand to more than one token, so a
    quantifier bound to the expansion's tail instead of the character: `a.?b`
    lost the zero-character branch and `a.+b` compiled as a POSSESSIVE repeat
    that matches nothing. Both compiled cleanly, so the re.compile guard on the
    translated pattern could never catch them."""
    translated = translate_pattern(pattern, 0)
    assert translated is not None, pattern
    want = [m.group() for m in re.finditer(pattern, subject)]
    got = [m.group().decode() for m in re.finditer(translated, subject.encode())]
    assert got == want, (pattern, subject)


def test_no_non_ascii_literal_reaches_tier_one_under_ignorecase():
    """CLASS-level and NOT vacuous, which the first attempt was.

    Round 5: quantifying over the right pattern alphabet but comparing against
    `{ch, lower, upper, casefold}` made the assertion pass for the whole
    alphabet by *skipping* — that subject set is exactly the relation round 3
    proved does not generate fold partners. The property that actually holds is
    simpler and has no escape hatch: under `re.I`, **no** non-ASCII literal is
    translated at all, in either literal form.
    """
    for span in (range(0x80, 0x0530), range(0x0530, 0x0700), range(0x13A0, 0x1400),
                 range(0x1E00, 0x2000), range(0x2100, 0x2500),
                 range(0x1D400, 0x1D420), range(0x10400, 0x10420),
                 range(0x1E900, 0x1E920)):
        for cp in span:
            ch = chr(cp)
            for pattern in (ch, "\\" + ch):
                assert translate_pattern(pattern, re.IGNORECASE) is None, (
                    hex(cp), pattern
                )
    # and the property is scoped to re.I only — without it they still translate
    assert translate_pattern("caf\u00e9", 0) is not None
    assert translate_pattern("\u6f22", 0) is not None


def test_known_flags_refuses_anything_not_reasoned_about():
    """Every flag `re` defines is either handled or refused — asserted over the
    flag SET, not over two hand-picked members. Round 2's `re.ASCII` defect had
    exactly the shape this catches: a flag admitted without its consequence
    being worked out."""
    handled = {re.IGNORECASE, re.DOTALL, re.MULTILINE, re.UNICODE, re.ASCII}
    for flag in re.RegexFlag:
        if flag in handled or flag == re.NOFLAG:
            continue
        assert translate_pattern("abc", flag) is None, flag


@pytest.mark.parametrize("cp", [0x00DF, 0x00C5, 0x0130, 0x03A3, 0x1E9E, 0x24B6])
def test_named_non_ascii_literals_under_ignorecase(cp):
    """The round-2 fix was one-directional: it stopped tier 1 matching where
    CPython does not, but left it *under*-matching for non-ASCII pattern
    characters, whose fold partners are not reachable from
    lower/upper/casefold. A miss is a wrong answer too, so the character is
    refused — and this test enumerates over PATTERNS, not just subjects, which
    is the gap that let the one-directional fix certify itself."""
    ch = chr(cp)
    translated = translate_pattern(ch, re.IGNORECASE)
    if translated is None:
        return
    for other in {ch, ch.lower(), ch.upper(), ch.casefold(), ch.title()}:
        assert bool(re.fullmatch(translated, other.encode("utf-8"))) is bool(
            re.fullmatch(ch, other, re.IGNORECASE)
        ), (ch, other)


@pytest.mark.parametrize(
    "pattern,subject",
    [(r"[abc]*", "a\U0001d54fb"), (r"x?", "caf\u00e9"), (r"[a-z]*", "caf\u00e9"),
     (r"\w*", "abc"), (r"a*", "\u03b1\u03b1\u03b1"), (r"(?:zz)*", "a\U0001d54fb")],
)
def test_zero_width_matches_land_only_on_character_positions(
    tmp_path, pattern, subject
):
    """Every guard before this one constrained matched CONTENT; a zero-width
    match constrains nothing. Tier 1 matched the empty string at each of the
    three continuation bytes inside an astral character — three spurious
    matches at one position, on the +0.0 MiB path, and `_as_text` could not
    see it because b"" decodes cleanly. Tier 2 had the mirror defect: an
    exclusive bound dropped the legal end-of-subject match."""
    path = tmp_path / "zw.txt"
    path.write_text(subject, encoding="utf-8")
    view = TextView.open(path)
    assert [m.span() for m in view._re_finditer(pattern)] == [
        m.span() for m in re.finditer(pattern, subject)
    ]


def test_ignorecase_fold_set_is_exactly_four_codepoints():
    """Enumerated over the whole of Unicode, not asserted from memory: FOUR
    non-ASCII codepoints participate in an ASCII-pattern `re.I` match. Round 1
    shipped three — U+0130 was missing, so `re.search("istanbul", "\u0130stanbul",
    re.I)` matched in CPython and not on tier 1."""
    participating = {
        cp
        for cp in range(0x80, 0x110000)
        for a in "abcdefghijklmnopqrstuvwxyz"
        if re.fullmatch(a, chr(cp), re.IGNORECASE)
    }
    assert participating == {0x0130, 0x0131, 0x017F, 0x212A}
    covered = {ord(c) for group in _FOLD_EXTRA.values() for c in group}
    assert covered == participating


@pytest.mark.parametrize(
    "pattern,subject",
    [("istanbul", "\u0130stanbul"), ("kelvin", "100\u212aelvin"),
     ("class", "cla\u017fs"), ("kirmak", "K\u0131rmak")],
)
def test_every_folding_codepoint_is_carried(pattern, subject):
    translated = translate_pattern(pattern, re.IGNORECASE)
    assert translated is not None
    assert bool(re.search(translated, subject.encode())) is bool(
        re.search(pattern, subject, re.IGNORECASE)
    )


@pytest.mark.parametrize("pattern,subject", [("\u00df", "SS"), ("\ufb01", "FI"),
                                             ("\u0149", "\u02bcN")])
def test_full_case_mapping_never_becomes_an_alternative(pattern, subject):
    """`str.upper()` is FULL case mapping and `re` folding is SIMPLE, so `ß`
    must not gain an "SS" alternative. A spurious match is worse than a miss:
    the caller cannot tell it apart from a real one."""
    assert not re.search(pattern, subject, re.IGNORECASE)
    translated = translate_pattern(pattern, re.IGNORECASE)
    if translated is not None:
        assert not re.search(translated, subject.encode())


def test_ascii_flag_narrows_ignorecase_too():
    """`re.A` is not only about `\\w \\d \\s \b`: it also stops IGNORECASE from
    reaching the fold partners, so emitting them under re.A is a spurious hit."""
    kelvin = "\u212a"
    assert re.fullmatch("k", kelvin, re.IGNORECASE)
    assert not re.fullmatch("k", kelvin, re.IGNORECASE | re.ASCII)
    translated = translate_pattern("k", re.IGNORECASE | re.ASCII)
    if translated is not None:
        assert not re.fullmatch(translated, kelvin.encode())


@pytest.mark.parametrize(
    "pattern", ["(?:.){0,200}NEEDLE", "(.)+", "(?:a.b)*", ".{0,1000}X"]
)
def test_quantified_dot_is_refused_through_a_group_too(pattern):
    """A one-character lookahead missed `(?:.){0,200}`, which reached tier 1 and
    measured 32.5x the original pattern. The predicate is deliberately
    over-broad: an unnecessary refusal costs a window, a missed one costs a
    sweep."""
    assert translate_pattern(pattern, 0) is None


@pytest.mark.parametrize("pattern", ["a.b", "[.]{2,3}", r"a\.b+"])
def test_unquantified_or_literal_dot_still_translates(pattern):
    """The over-broad predicate must not swallow a `.` that carries no
    quantifier, one inside a class, or an escaped literal."""
    assert translate_pattern(pattern, 0) is not None


def test_tier2_scan_restarts_at_the_owned_position(tmp_path, monkeypatch):
    """Filtering tier-2 output is not the same as restarting its scan: a greedy
    match found before the owned position and discarded differs from the match a
    correctly-positioned scan produces. Randomised over tiny windows, which is
    where the boundary cases actually live."""
    import random

    monkeypatch.setattr(TextView, "WINDOW_BYTES", 256)
    monkeypatch.setattr(TextView, "OVERLAP_BYTES", 64)
    rng = random.Random(7)
    alphabet = "abc de\tf\ngh\u00e9 \uc784 \U0001F600"
    for trial in range(40):
        text = "".join(rng.choice(alphabet) for _ in range(rng.randint(200, 900)))
        path = tmp_path / f"r{trial}.txt"
        path.write_text(text, encoding="utf-8")
        view = TextView.open(path)
        for pattern in (r"\w+\s+\w+", r"[^a]+", r"\S+"):
            got = [m.span() for m in view._re_finditer(pattern)]
            want = [m.span() for m in re.finditer(pattern, text)]
            assert got == want, (pattern, trial)


def test_ignorecase_covers_the_folding_codepoints():
    """CPython's re uses SIMPLE folding, so a small fixed set of non-ASCII
    codepoints participate in an ASCII-pattern re.I match — FOUR of them, as the
    enumeration above proves. Every one has to be carried or tier 1 silently
    under-matches."""
    for pattern, text in [("kelvin", "100Kelvin"), ("class", "cla\u017fs"),
                          ("kirmak", "K\u0131rmak"),
                          ("istanbul", "\u0130stanbul")]:
        translated = translate_pattern(pattern, re.IGNORECASE)
        assert translated is not None
        want = bool(re.search(pattern, text, re.IGNORECASE))
        got = bool(re.search(translated, text.encode("utf-8")))
        assert got == want, (pattern, text)


# --------------------------------------------------------------------------- #
# the re protocol, all tiers
# --------------------------------------------------------------------------- #
def test_finditer_matches_str_and_stays_on_tier1(view):
    escape_reset()
    got = [(m.start(), m.end(), m.group()) for m in view._re_finditer("leukemia")]
    want = [(m.start(), m.end(), m.group()) for m in re.finditer("leukemia", SAMPLE)]
    assert got == want
    assert escape_snapshot() == {}
    assert any(k.startswith("tier1") for k in search_tier_snapshot())


def test_tier2_window_serves_a_refused_pattern_exactly(view):
    """``\\S+`` is refused by the translator, so this exercises the windowed
    path — and it must still equal ``str``."""
    escape_reset()
    pattern = r"\S+data"
    got = [(m.start(), m.end(), m.group()) for m in view._re_finditer(pattern)]
    want = [(m.start(), m.end(), m.group()) for m in re.finditer(pattern, SAMPLE)]
    assert got == want
    assert any(k.startswith("tier2") for k in search_tier_snapshot())


def test_tier2_windows_are_smaller_than_the_subject(view, monkeypatch):
    """Force many windows so boundary handling is actually exercised."""
    monkeypatch.setattr(TextView, "WINDOW_BYTES", 4096)
    monkeypatch.setattr(TextView, "OVERLAP_BYTES", 512)
    pattern = r"\S+data"
    got = [(m.start(), m.end()) for m in view._re_finditer(pattern)]
    want = [(m.start(), m.end()) for m in re.finditer(pattern, SAMPLE)]
    assert got == want


def test_corpus_max_document_is_stated_against_the_overlap():
    """The disclosure's load-bearing number must be the corpus MAXIMUM, not its
    mean — three review rounds caught an average or a stale constant standing in
    for a bound. Pinned here so a re-measurement that changes it fails loudly."""
    doc = TextView._windowed_finditer.__doc__ or ""
    assert "7,094,565" in doc, "the corpus maximum document size must be stated"
    assert "27x the overlap" in doc or "27x" in doc
    assert "VIOLATES" in doc, "a violated precondition must say so in the word"
    assert "mean" in doc, "the mean/maximum distinction must survive edits"


def test_tier2_bound_is_stated_and_is_the_overlap():
    """Tier 2's exactness bound is the OVERLAP, and the docstring must say so.

    Window growth was built in ABR rounds 6-8 to widen this bound and was
    removed in round 10: round 9's E6b priced it at 1,097 s for one task
    against 281 s without it, tripping G3. The bound went back to the overlap
    and the disclosure has to go back with it — an unstated narrowing is worse
    than a narrow one.
    """
    doc = TextView._windowed_finditer.__doc__ or ""
    assert "OVERLAP_BYTES" in doc
    assert "silent" in doc.lower()
    # the live bound is the overlap; `2 x WINDOW_BYTES` may only appear as
    # the history of what growth briefly bought before it was removed
    assert "removed in round 10" in doc


def test_tier2_matches_longer_than_the_overlap_are_not_fragmented(
    tmp_path, monkeypatch
):
    """The overlap lets a match that STARTS in the owned region FINISH; it is
    not licence to scan that region twice. Round 1 rescanned it as a fresh
    subject, so a match crossing a boundary was emitted whole by one window and
    again as a tail fragment by the next — a spurious extra match, not a
    duplicate a span set can absorb.

    The round-1 test could not see this: every match it used was far shorter
    than the overlap, so none ever crossed a boundary. These are ~1500 chars
    against a 512-byte overlap.
    """
    monkeypatch.setattr(TextView, "WINDOW_BYTES", 4096)
    monkeypatch.setattr(TextView, "OVERLAP_BYTES", 512)
    text = ("é" + "b" * 1500) * 12
    path = tmp_path / "long.txt"
    path.write_text(text, encoding="utf-8")
    v = TextView.open(path)
    pattern = r"[^é]+"  # negated class -> refused by tier 1 -> tier 2
    got = [(m.start(), m.end()) for m in v._re_finditer(pattern)]
    want = [(m.start(), m.end()) for m in re.finditer(pattern, text)]
    assert got == want


def test_edge_sensitive_patterns_leave_the_window_for_the_whole_subject(
    tmp_path, monkeypatch
):
    """A window is a DIFFERENT subject: `^` fires at every window start and a
    lookaround sees truncated context. Growing the window cannot repair that, so
    these patterns take tier 3, where the answer is exactly today's."""
    monkeypatch.setattr(TextView, "WINDOW_BYTES", 4096)
    monkeypatch.setattr(TextView, "OVERLAP_BYTES", 512)
    text = ("data line here é\n" * 900)
    path = tmp_path / "lines.txt"
    path.write_text(text, encoding="utf-8")
    v = TextView.open(path)
    for pattern, flags in [(r"^data\S*", re.MULTILINE), (r"(?<=line )here", 0)]:
        got = [(m.start(), m.end()) for m in v._re_finditer(pattern, flags)]
        want = [(m.start(), m.end()) for m in re.finditer(pattern, text, flags)]
        assert got == want, pattern


def test_undispatched_re_entry_points_escape_instead_of_raising(view):
    """WO003's shim materialised for anything it could not serve lazily. Round 1
    replaced it with a dispatch table and let everything else through to the raw
    C function, which receives a TextView and raises TypeError — so
    `re.match(p, ctx)` CRASHED the guest where the predecessor merely escaped.

    `sub`/`subn` are the sharp case: their subject is the third positional, so
    the two-positional binding never even saw it.
    """
    import rlm.context_view as cvmod

    proxy = cvmod.make_coercing_import()("re", {}, {}, [], 0)
    escape_reset()
    assert proxy.match("the", view).group() == "the"
    assert proxy.fullmatch(".*", view, re.DOTALL) is not None
    assert proxy.sub("leukemia", "X", view).startswith("the X")
    assert proxy.subn("leukemia", "Y", view)[1] == re.subn("leukemia", "Y", SAMPLE)[1]
    assert proxy.compile("the").match(view).group() == "the"
    assert proxy.compile("leukemia").sub("Z", view) == re.sub("leukemia", "Z", SAMPLE)
    # every one of them is COUNTED, or the escape census under-reports
    assert set(escape_snapshot()) >= {
        "re.match", "re.fullmatch", "re.sub", "re.subn",
        "re.Pattern.match", "re.Pattern.sub",
    }


def test_re_split_maxsplit_is_not_bound_to_flags(view):
    """`re.split(pattern, string, maxsplit, flags)` and the view method take
    their arguments in different orders; round 1 forwarded positionally."""
    got = view._re_split(r"\s+", 0, 2)
    assert got == re.split(r"\s+", SAMPLE, maxsplit=2)


def test_bounded_repeat_over_dot_stays_linear(tmp_path):
    """A COST test, not a correctness test — because this failure mode has now
    escaped a green correctness suite twice.

    `.` expands to more than one token, so a bounded repeat over it (`.{0,200}`
    is ordinary in this corpus) hands the engine a nested quantifier unless the
    expansion is atomic. Non-atomic, a real ladder row ran 1278 s against a
    900 s limit; the textbook 9-branch alternation before it ran 24 minutes.
    Atomic, the same scan is sub-second — so a generous ceiling separates the
    classes without being flaky.
    """
    import time

    unit = "Robbie went to the café and read 0123456789 lines of text here. "
    text = unit * ((4 << 20) // len(unit.encode()))
    path = tmp_path / "big.txt"
    path.write_text(text, encoding="utf-8")
    v = TextView.open(path)

    started = time.perf_counter()
    got = sum(1 for _ in v._re_finditer(r"Robbie.{0,200}"))
    elapsed = time.perf_counter() - started

    assert got == len(re.findall(r"Robbie.{0,200}", text))
    assert elapsed < 10.0, f"bounded repeat took {elapsed:.1f}s — backtracking"
    # It is served on TIER 2 by design: the original pattern on a window costs
    # ~1/30th of the translated one, so the refusal is the fast path here.
    assert any(k.startswith("tier2") for k in search_tier_snapshot())


def test_search_and_findall_match_str(view):
    assert view._re_search("café").span() == re.search("café", SAMPLE).span()
    assert view._re_search("absent-token") is None
    got = view._re_findall(r"(leukemia|trial)")
    want = re.findall(r"(leukemia|trial)", SAMPLE)
    assert got == want


def test_match_group_returns_real_str_and_offsets_are_char_offsets(view):
    m = view._re_search(r"caf(é)")
    want = re.search(r"caf(é)", SAMPLE)
    assert isinstance(m.group(0), str)
    assert m.group(0) == want.group(0)
    assert m.group(1) == want.group(1)
    assert m.span() == want.span()
    assert SAMPLE[m.start():m.end()] == m.group(0)


def test_match_string_is_an_escape_and_is_logged(view):
    escape_reset()
    m = view._re_search("leukemia")
    assert m.string == SAMPLE
    assert "match.string" in escape_snapshot()


# --------------------------------------------------------------------------- #
# escape hatch — never a silent wrong answer
# --------------------------------------------------------------------------- #
def test_unknown_str_method_falls_back_and_is_logged(view):
    escape_reset()
    assert view.upper() == SAMPLE.upper()
    assert "attr:upper" in escape_snapshot()


def test_str_conversion_is_logged(view):
    escape_reset()
    assert str(view) == SAMPLE
    assert escape_snapshot().get("__str__") == 1


# --------------------------------------------------------------------------- #
# the interception shim
# --------------------------------------------------------------------------- #
def _run_guest(code: str, view: TextView) -> dict:
    builtins_dict = dict(__builtins__) if isinstance(__builtins__, dict) else dict(
        __builtins__.__dict__
    )
    builtins_dict["__import__"] = make_coercing_import()
    ns = {"__builtins__": builtins_dict, "context": view}
    exec(compile(code, "<guest>", "exec"), ns)  # noqa: S102 - that is the point
    return ns


def test_guest_re_is_served_from_the_mapping_without_escaping(view):
    escape_reset()
    ns = _run_guest(
        "import re\n"
        "ms = list(re.finditer('leukemia', context))\n"
        "starts = [m.start() for m in ms]\n"
        "p = re.compile(r'trial')\n"
        "n = len(list(p.finditer(context)))\n"
        "hit = re.search('caf\\u00e9', context)\n",
        view,
    )
    assert ns["starts"] == [m.start() for m in re.finditer("leukemia", SAMPLE)]
    assert ns["n"] == len(re.findall("trial", SAMPLE))
    assert ns["hit"].start() == SAMPLE.find("café")
    assert escape_snapshot() == {}


def test_shim_passes_non_view_arguments_through_untouched(view):
    ns = _run_guest(
        "import re\n"
        "plain = [m.group() for m in re.finditer(r'\\d+', 'a12b345')]\n",
        view,
    )
    assert ns["plain"] == ["12", "345"]


def test_shim_leaves_classes_unwrapped(view):
    """Wrapping classes broke ``except``/``isinstance``/subclassing in WO003 to
    fix a path with zero corpus occurrences; that revert is pinned here."""
    ns = _run_guest(
        "import json\n"
        "err = json.JSONDecodeError\n"
        "try:\n"
        "    json.loads('{oops')\n"
        "    caught = False\n"
        "except json.JSONDecodeError:\n"
        "    caught = True\n",
        view,
    )
    assert ns["caught"] is True
    assert isinstance(ns["err"], type)


def test_other_c_level_modules_still_materialise_and_log(view):
    escape_reset()
    ns = _run_guest("import json\nout = json.dumps(context)\n", view)
    assert ns["out"].startswith('"the leukemia')
    assert any(k.startswith("c-level:json") for k in escape_snapshot())


# --------------------------------------------------------------------------- #
# integration with LocalREPL
# --------------------------------------------------------------------------- #
def test_localrepl_binds_a_view_for_a_viewref(tmp_path):
    from rlm.environments.local_repl import LocalREPL

    path = tmp_path / "ctx.txt"
    path.write_text(SAMPLE, encoding="utf-8")
    env = LocalREPL(context_mode="view")
    env.add_context(ViewRef(str(path), len(SAMPLE)))
    assert isinstance(env.locals["context_0"], TextView)
    assert env.locals["context"] is env.locals["context_0"]
    env.execute_code(
        "import re\n"
        "hits = [m.start() for m in re.finditer('leukemia', context)]\n"
    )
    assert env.locals["hits"] == [m.start() for m in re.finditer("leukemia", SAMPLE)]


def test_mapped_bytes_exposes_a_buffer_re_can_scan(tmp_path):
    path = tmp_path / "b.bin"
    path.write_bytes(b"alpha beta gamma")
    mapped = MappedBytes(path)
    try:
        assert len(mapped) == 16
        assert bytes(mapped[0:5]) == b"alpha"
        assert re.search(rb"beta", mapped.buffer).start() == 6
    finally:
        mapped.close()
