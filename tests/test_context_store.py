"""Property + unit tests for the WO003 context store (RLM Lab, BR002-WO003).

E2 gate: view op ≡ str op on identical content (random slices, finds,
containment, boundary offsets) + escape-hatch logging + store roundtrip.
"""

import random

import pytest

from rlm.context_store import (
    ContextStore,
    LazyStr,
    PathRef,
    RecorderStr,
    StoreRef,
    escape_reset,
    escape_snapshot,
)

# Non-ASCII on purpose: the corpus is 99.7% ASCII but every file carries emoji
# (E1a), so correctness must hold on UCS-4 content.
SAMPLE = (
    "The quick brown fox 🦊 jumps over the lazy dog.\n"
    "Document <id=42> τίτλος — naïve UTF-32 first.\n"
    "  padded line with trailing spaces   \n"
) * 50
EMPTY = ""


@pytest.fixture()
def store(tmp_path):
    return ContextStore(tmp_path / "store")


@pytest.fixture()
def view(store) -> LazyStr:
    handle = store.ingest_text(SAMPLE)
    return store.open(handle)


def test_ingest_is_content_addressed(store):
    h1 = store.ingest_text(SAMPLE)
    h2 = store.ingest_text(SAMPLE)
    assert h1 == h2
    assert store.segment_path(h1).stat().st_size == len(SAMPLE) * 4


def test_ref_roundtrip(store, tmp_path):
    src = tmp_path / "doc.txt"
    src.write_text(SAMPLE, encoding="utf-8")
    entry = store.ingest_file(src, name="doc")
    ref = StoreRef.from_index(store.root, "doc")
    assert ref.chars == len(SAMPLE) == entry["chars"]
    assert ref.context_char_count == len(SAMPLE)
    v = ref.open()
    assert v[: len(SAMPLE)] == SAMPLE


def test_len_and_index(view):
    assert len(view) == len(SAMPLE)
    for i in (0, 1, 20, len(SAMPLE) - 1, -1, -len(SAMPLE)):
        assert view[i] == SAMPLE[i]
    with pytest.raises(IndexError):
        view[len(SAMPLE)]


def test_random_slices_match_str(view):
    rng = random.Random(0)
    n = len(SAMPLE)
    for _ in range(300):
        a = rng.randint(-n - 2, n + 2)
        b = rng.randint(-n - 2, n + 2)
        step = rng.choice([None, 1, 2, 3, -1, -2])
        s = slice(a, b, step)
        assert view[s] == SAMPLE[s], f"slice {s}"


def test_boundary_slices(view):
    n = len(SAMPLE)
    for s in (slice(0, 0), slice(n, n), slice(0, n), slice(None), slice(n - 1, n),
              slice(-1, None), slice(None, None, -1), slice(5, 5, 2)):
        assert view[s] == SAMPLE[s]


def test_find_rfind_index_count(view):
    rng = random.Random(1)
    needles = ["fox", "🦊", "naïve", "line", "\n", "zzz-not-there", "e", "  ",
               "τίτλος", "dog.\nDocument"]
    n = len(SAMPLE)
    for needle in needles:
        assert view.find(needle) == SAMPLE.find(needle)
        assert view.rfind(needle) == SAMPLE.rfind(needle)
        assert view.count(needle) == SAMPLE.count(needle)
        for _ in range(25):
            a = rng.randint(-n, n)
            b = rng.randint(-n, n)
            assert view.find(needle, a, b) == SAMPLE.find(needle, a, b), (needle, a, b)
            assert view.rfind(needle, a, b) == SAMPLE.rfind(needle, a, b), (needle, a, b)
            assert view.count(needle, a, b) == SAMPLE.count(needle, a, b), (needle, a, b)
    with pytest.raises(ValueError):
        view.index("zzz-not-there")
    assert view.index("fox") == SAMPLE.index("fox")


def test_census_native_ops_no_escape(view):
    """The 7 census ops run lazily — no escape events."""
    escape_reset()
    assert len(view) == len(SAMPLE)
    assert view[10:40] == SAMPLE[10:40]
    assert view.find("fox") == SAMPLE.find("fox")
    assert view.rfind("fox") == SAMPLE.rfind("fox")
    assert view.count("line") == SAMPLE.count("line")
    assert view.lower() == SAMPLE.lower()
    assert view.split("\n") == SAMPLE.split("\n")
    assert not any(k.startswith("escape:") for k in escape_snapshot())
    escape_reset()


def test_non_census_ops_escape_but_stay_correct(view):
    """Everything outside the census routes through the logged escape hatch —
    correct results, measured cost."""
    escape_reset()
    assert ("fox" in view) == ("fox" in SAMPLE)
    assert ("zzz" in view) == ("zzz" in SAMPLE)
    assert view.startswith("The quick") == SAMPLE.startswith("The quick")
    assert view.strip() == SAMPLE.strip()
    assert view.upper() == SAMPLE.upper()
    assert view.splitlines() == SAMPLE.splitlines()
    assert view.replace("fox", "cat") == SAMPLE.replace("fox", "cat")
    assert view.index("fox") == SAMPLE.index("fox")
    assert "".join(view) == SAMPLE
    snap = escape_snapshot()
    assert snap.get("escape:__contains__") == 2
    assert snap.get("escape:getattr:startswith") == 1
    assert snap.get("escape:getattr:strip") == 1
    assert snap.get("escape:__iter__") == 1
    escape_reset()


def test_bool_and_eq_protocol_safety(view):
    escape_reset()
    assert bool(view) is True
    assert view == SAMPLE
    assert not (view == SAMPLE + "x")
    assert view != SAMPLE[:-1]
    # eq is the documented lazy deviation — it must NOT materialize
    assert not any(k.startswith("escape:") for k in escape_snapshot())
    escape_reset()


def test_returns_are_real_str(view):
    assert type(view[0:10]) is str
    assert type(view.lower()) is str
    assert all(type(part) is str for part in view.split("\n")[:5])


def test_escape_hatch_logging(view):
    escape_reset()
    _ = str(view)
    snap = escape_snapshot()
    assert snap.get("escape:__str__") == 1
    # unknown method -> getattr escape, then delegates natively
    escape_reset()
    assert view.title() == SAMPLE.title()
    assert escape_snapshot().get("escape:getattr:title") == 1
    escape_reset()


def test_empty_segment(store):
    handle = store.ingest_text(EMPTY)
    # zero-length mmap is invalid; empty context is out of corpus scope —
    # ingest still content-addresses it, open() is not required to succeed.
    assert handle


def test_recorder_str_counts():
    RecorderStr.reset()
    r = RecorderStr("alpha beta alpha")
    assert r.count("alpha") == 2
    assert r[0:5] == "alpha"
    assert "beta" in r
    assert len(r) == 16
    snap = RecorderStr.snapshot()
    assert snap["count"] == 1
    assert snap["__getitem__[slice]"] == 1
    assert snap["__contains__"] == 1
    assert snap["__len__"] == 1
    RecorderStr.reset()
    assert RecorderStr.snapshot() == {}
def test_c_level_coercion_shim(tmp_path):
    """C-level exact-str demands materialize through the logged escape hatch
    (FR002) instead of raising TypeError — the E3 boundary finding."""
    import re as real_re

    from rlm.context_store import make_coercing_import

    store = ContextStore(tmp_path / "store")
    text = "alpha 42 beta 7 gamma\n" * 100
    handle = store.ingest_text(text)
    view = store.open(handle)

    # Raw C call rejects the view (this is the boundary itself).
    with pytest.raises(TypeError):
        real_re.finditer(r"\d+", view)

    guest_import = make_coercing_import()
    re_shim = guest_import("re")
    escape_reset()
    hits = [m.start() for m in re_shim.finditer(r"\d+", view)]
    assert hits == [m.start() for m in real_re.finditer(r"\d+", text)]
    assert escape_snapshot().get("escape:c-level:re.finditer") == 1

    # compiled patterns carry their own C methods — the proxy rides along
    escape_reset()
    pat = re_shim.compile(r"beta")
    assert [m.start() for m in pat.finditer(view)] == [
        m.start() for m in real_re.compile(r"beta").finditer(text)
    ]
    assert escape_snapshot().get("escape:c-level:re.compile().finditer") == 1

    # non-LazyStr args pass through untouched
    escape_reset()
    assert re_shim.findall(r"\d+", "a1 b22") == real_re.findall(r"\d+", "a1 b22")
    assert escape_snapshot() == {}
    escape_reset()
def test_container_nested_coercion_is_a_known_limitation(tmp_path):
    """Coercion is one level deep: a view nested inside a container still
    reaches the C encoder. Pinned rather than fixed — recursive coercion would
    walk every argument of every call (ABR 260731 finding, scoped)."""
    from rlm.context_store import make_coercing_import

    store = ContextStore(tmp_path / "store")
    view = store.open(store.ingest_text("payload " * 50))
    guest_import = make_coercing_import()

    escape_reset()
    with pytest.raises(TypeError):
        guest_import("json").dumps({"ctx": view})
    assert escape_snapshot() == {}       # and it is NOT counted as façade cost
    escape_reset()


def test_int_index_storm_is_logged(tmp_path):
    """Per-character walking raises nothing; without a counter it would be
    façade cost the census cannot see (ABR 260731 finding)."""
    store = ContextStore(tmp_path / "store")
    view = store.open(store.ingest_text("x" * 200_000))
    escape_reset()
    for i in range(LazyStr._INT_INDEX_STORM - 1):
        view[i]
    assert escape_snapshot() == {}                      # below threshold: quiet
    view[LazyStr._INT_INDEX_STORM - 1]
    assert escape_snapshot().get("escape:int-index-storm") == 1
    escape_reset()


def test_view_to_view_equality(tmp_path):
    """Two views over the same content must compare equal — the identity
    fallback answered False silently (ABR 260731 finding)."""
    store = ContextStore(tmp_path / "store")
    text = "alpha beta gamma " * 100
    a, b = store.open(store.ingest_text(text)), store.open(store.ingest_text(text))
    c = store.open(store.ingest_text(text + "x"))
    escape_reset()
    assert a == b and not (a == c) and a == text
    assert escape_snapshot() == {}                      # still lazy


def test_needle_range_sweep_matches_str(tmp_path):
    """Full (needle, start, end) sweep against `str`. The earlier version tested
    a hand-picked tuple set that happened to agree at `("",10)` while
    `find("",11)` returned 10 instead of -1 (ABR 260731 R2-2): `start` is NOT
    clamped down to len, only `end` is."""
    import itertools

    store = ContextStore(tmp_path / "store")
    text = "abcdefghij"
    v = store.open(store.ingest_text(text))
    vals = [None, -99, -11, -3, 0, 1, 5, 10, 11, 99]

    checked = 0
    for needle in ("", "a", "cd", "zz"):
        for s, e in itertools.product(vals, vals):
            args = [needle] if (s is None and e is None) else (
                [needle, s, e] if e is not None else [needle, s]
            )
            checked += 1
            assert v.find(*args) == text.find(*args), args
            assert v.rfind(*args) == text.rfind(*args), args
            assert v.count(*args) == text.count(*args), args
    assert checked >= 400


def test_classes_pass_through_unwrapped(tmp_path):
    """Classes must NOT be proxied. Coercing constructor args broke `except`,
    `isinstance` and subclassing — a silent-wrong-answer trade for a path with
    zero corpus occurrences (ABR 260731 R2-1, reverted)."""
    import ast
    import json as real_json

    from rlm.context_store import make_coercing_import

    guest_import = make_coercing_import()
    j, a = guest_import("json"), guest_import("ast")

    # except must still catch
    try:
        real_json.loads("{bad")
    except j.JSONDecodeError:
        pass

    # isinstance must not silently flip
    node = ast.parse("x = 1").body[0]
    assert isinstance(node, a.Assign) is True

    # subclassing must still work
    class E(j.JSONEncoder):
        pass

    assert issubclass(E, real_json.JSONEncoder)

    # and the function path is still coerced
    store = ContextStore(tmp_path / "store")
    view = store.open(store.ingest_text("digits 4242 here\n" * 30))
    escape_reset()
    hits = list(guest_import("re").finditer(r"\d+", view))
    assert len(hits) == 30
    assert escape_snapshot().get("escape:c-level:re.finditer") == 1
    escape_reset()


def test_local_repl_externalized_bind(tmp_path):
    """E3 integration: StoreRef payload binds a LazyStr view in the guest."""
    from rlm.environments.local_repl import LocalREPL

    store = ContextStore(tmp_path / "store")
    text = "alpha beta 🦊 gamma\n" * 200
    src = tmp_path / "doc.txt"
    src.write_text(text, encoding="utf-8")
    store.ingest_file(src, name="doc")
    ref = StoreRef.from_index(store.root, "doc")

    env = LocalREPL(context_mode="externalized")
    env.add_context(ref)
    assert type(env.locals["context_0"]).__name__ == "LazyStr"
    assert env.locals["context"] is env.locals["context_0"]

    res = env.execute_code(
        "n = len(context)\n"
        "hit = context.find('beta')\n"
        "sl = context[6:10]\n"
        "has = '🦊' in context\n"
        "print(n, hit, sl, has)"
    )
    assert res.stderr.strip() == ""
    n, hit, sl, has = res.stdout.split()
    assert int(n) == len(text)
    assert int(hit) == text.find("beta")
    assert sl == text[6:10]
    assert has == "True"

    # scaffold restore keeps `context` aliased to the view across turns
    env.execute_code("context = 'clobbered'")
    res2 = env.execute_code("print(len(context))")
    assert int(res2.stdout.strip()) == len(text)




def test_pathref_lane_binds_stock_str(tmp_path):
    """Cheap-alternative control: PathRef passes a path; the guest does the
    stock f.read() — real str, no store, no coercion shim, no escapes."""
    import builtins

    from rlm.environments.local_repl import LocalREPL

    text = "alpha beta 🦊 gamma\n" * 300
    src = tmp_path / "ctx.txt"
    src.write_text(text, encoding="utf-8")
    ref = PathRef(str(src), src.stat().st_size)
    assert ref.context_char_count == src.stat().st_size  # byte bound, documented

    escape_reset()
    env = LocalREPL(context_mode="path")
    env.add_context(ref)

    ctx = env.locals["context_0"]
    assert type(ctx) is str                      # exact str, not a view
    assert ctx == text
    assert env.locals["context"] is ctx
    # stock __import__ must survive: no coercion shim in this lane
    assert env.globals["__builtins__"]["__import__"] is builtins.__import__

    res = env.execute_code(
        "import re\n"
        "n = len(context)\n"
        "hits = len(list(re.finditer(r'beta', context)))\n"
        "print(n, hits, context[6:10])"
    )
    assert res.stderr.strip() == ""
    n, hits, sl = res.stdout.split()
    assert int(n) == len(text)
    assert int(hits) == text.count("beta")
    assert sl == text[6:10]
    assert escape_snapshot() == {}                # no façade, so no façade cost
