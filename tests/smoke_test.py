"""
smoke_test.py — chunk fragmentation fix, logic-only verification.

Synthetic data, stubbed rerank(), no live Chroma or Gemini calls. Verifies
_fragmented_ids(), _resolve_fragments(), and their wiring into
_guaranteed_slot_rerank_merge() — nothing about retrieval quality, that's
what the targeted/full-suite passes are for.

Usage: python smoke_test.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import retrieval_pipeline as rp


# ── Fakes ────────────────────────────────────────────────────────────────

class FakeCollection:
    """
    Mimics db.collection.get(where=..., include=...) against an in-memory
    corpus. `corpus_data`: {(corpus, technique_id): [(meta, doc), ...]}.
    Tracks call count so tests can assert caching/targeting behavior, not
    just final output.
    """
    def __init__(self, corpus_data: dict):
        self.corpus_data = corpus_data
        self.get_calls = 0

    def get(self, where=None, include=None):
        self.get_calls += 1
        if where is None:
            # full-collection pull, as used by _fragmented_ids()
            metas, docs = [], []
            for chunks in self.corpus_data.values():
                for meta, doc in chunks:
                    metas.append(meta)
                    docs.append(doc)
            return {"metadatas": metas, "documents": docs}

        # targeted pull, as used by _resolve_fragments()
        conds = {c["$eq"] for cond in where["$and"] for c in [list(cond.values())[0]]}
        corpus = where["$and"][0]["corpus"]["$eq"]
        tid = where["$and"][1]["technique_id"]["$eq"]
        chunks = self.corpus_data.get((corpus, tid), [])
        return {
            "metadatas": [m for m, _ in chunks],
            "documents": [d for _, d in chunks],
        }


class FakeDB:
    def __init__(self, corpus_data: dict):
        self.collection = FakeCollection(corpus_data)


def stub_rerank(query, candidates, top_k):
    """
    Deterministic stand-in for the real cross-encoder: score = document
    length. Lets tests assert "the merged, longer text won" without caring
    about real relevance — that's not what this test is checking.
    """
    scored = [(meta, doc, float(len(doc))) for meta, doc in candidates]
    scored.sort(key=lambda x: -x[2])
    return scored[:top_k]


# ── Test data ────────────────────────────────────────────────────────────

CORPUS = {
    ("mitre", "T1621"): [
        ({"corpus": "mitre", "technique_id": "T1621"}, "T1621 chunk one, short."),
        ({"corpus": "mitre", "technique_id": "T1621"}, "T1621 chunk two, also short."),
        ({"corpus": "mitre", "technique_id": "T1621"}, "T1621 chunk three, longest of the three."),
    ],
    ("mitre", "T1111"): [
        ({"corpus": "mitre", "technique_id": "T1111"}, "T1111 chunk one."),
        ({"corpus": "mitre", "technique_id": "T1111"}, "T1111 chunk two, slightly longer."),
    ],
    ("mitre", "T1078"): [
        ({"corpus": "mitre", "technique_id": "T1078"}, "T1078 is single-chunk, never fragmented."),
    ],
}


def fresh_db():
    return FakeDB(CORPUS)


# ── Tests ────────────────────────────────────────────────────────────────

def test_fragmented_ids_correct():
    db = fresh_db()
    ids = rp._fragmented_ids(db)
    assert ids == {"T1621", "T1111"}, f"expected {{T1621, T1111}}, got {ids}"
    print("PASS: _fragmented_ids identifies exactly the >1-chunk techniques")


def test_fragmented_ids_cached():
    db = fresh_db()
    rp._FRAGMENTED_IDS = None  # reset module cache between tests
    rp._fragmented_ids(db)
    calls_after_first = db.collection.get_calls
    rp._fragmented_ids(db)
    assert db.collection.get_calls == calls_after_first, "second call should not re-query"
    print("PASS: _fragmented_ids caches after first call")


def test_resolve_fragments_passthrough_for_single_chunk():
    db = fresh_db()
    rp._FRAGMENTED_IDS = None
    candidates = [({"corpus": "mitre", "technique_id": "T1078"}, "T1078 is single-chunk, never fragmented.")]
    calls_before = db.collection.get_calls
    resolved = rp._resolve_fragments(db, candidates)
    assert resolved == candidates, "non-fragmented candidate must pass through unchanged"
    # one extra call expected: _fragmented_ids' own full-collection pull
    print("PASS: single-chunk technique passes through untouched")


def test_resolve_fragments_merges_all_chunks():
    db = fresh_db()
    rp._FRAGMENTED_IDS = None
    # Simulate what the pool actually contains: only ONE of T1621's three
    # chunks surfaced in this candidate list — the bug this fix targets.
    candidates = [({"corpus": "mitre", "technique_id": "T1621"}, "T1621 chunk one, short.")]
    resolved = rp._resolve_fragments(db, candidates)
    assert len(resolved) == 1
    meta, doc = resolved[0]
    assert meta["fragment_count"] == 3, f"expected 3 fragments merged, got {meta.get('fragment_count')}"
    assert "chunk one" in doc and "chunk two" in doc and "chunk three" in doc, \
        "merged doc must contain all three fragments' text"
    print("PASS: fragmented technique resolves to all 3 chunks, concatenated")

def test_resolve_fragments_dedupes_repeated_technique_in_pool():
    db = fresh_db()
    rp._FRAGMENTED_IDS = None
    rp._fragmented_ids(db)  # pre-warm — isolates the fragment-fetch dedup
                             # being tested here from the one-time cache-
                             # population cost already covered by
                             # test_fragmented_ids_cached.
    candidates = [
        ({"corpus": "mitre", "technique_id": "T1621"}, "T1621 chunk one, short."),
        ({"corpus": "mitre", "technique_id": "T1621"}, "T1621 chunk two, also short."),
    ]
    calls_before = db.collection.get_calls
    resolved = rp._resolve_fragments(db, candidates)
    calls_after = db.collection.get_calls
    assert len(resolved) == 1, "repeated technique_id in one pool must collapse to one resolved entry"
    assert calls_after - calls_before == 1, "must fetch all chunks only ONCE even if seen twice in the pool"
    print("PASS: repeated fragmented technique in one pool fetched only once")


def test_merge_uses_concatenated_text_not_partial_chunk():
    """
    Integration check: with rerank stubbed to reward LENGTH, a fragmented
    technique represented by only its shortest chunk in the raw pool must
    still win on its FULL concatenated length, not the short fragment's.
    This is the actual bug q005/q015 hit — proving the fix closes it.
    """
    db = fresh_db()
    rp._FRAGMENTED_IDS = None
    rp.rerank = stub_rerank  # monkeypatch the module-level import

    dense_candidates = [
        ({"corpus": "mitre", "technique_id": "T1078"}, "T1078 is single-chunk, never fragmented."),
    ]
    subquery_pools = [{
        "sub_query_text": "fake sub-query",
        "candidates": [
            # only T1621's SHORTEST chunk surfaced here — pre-fix, this is
            # all the reranker would ever see for T1621.
            ({"corpus": "mitre", "technique_id": "T1621"}, "T1621 chunk one, short."),
        ],
    }]

    result = rp._guaranteed_slot_rerank_merge(db, "raw query", dense_candidates, subquery_pools, k=2)
    t1621_result = next((r for r in result if r[0]["technique_id"] == "T1621"), None)
    assert t1621_result is not None, "T1621 should win its guaranteed slot"
    _, doc, _ = t1621_result
    assert "chunk one" in doc and "chunk two" in doc and "chunk three" in doc, \
        "winning T1621 candidate must be the full concatenated text, not the lone short fragment"
    print("PASS: guaranteed-slot merge scores and returns the FULL fragmented text")


# ── Runner ───────────────────────────────────────────────────────────────

TESTS = [
    test_fragmented_ids_correct,
    test_fragmented_ids_cached,
    test_resolve_fragments_passthrough_for_single_chunk,
    test_resolve_fragments_merges_all_chunks,
    test_resolve_fragments_dedupes_repeated_technique_in_pool,
    test_merge_uses_concatenated_text_not_partial_chunk,
]

if __name__ == "__main__":
    failures = 0
    for test in TESTS:
        try:
            test()
        except AssertionError as e:
            failures += 1
            print(f"FAIL: {test.__name__}: {e}")
    print(f"\n{len(TESTS) - failures}/{len(TESTS)} passed")
    sys.exit(1 if failures else 0)