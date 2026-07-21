"""
retrieval_pipeline.py — retrieve_for_route()

ARCHITECTURE, third revision (Jul 21, 2026 — see below for what changed
from the Jul 2026 merge+rerank version and why):

  Step 1 — exact match: unchanged since the original version. Literal
  CVE/technique IDs in the raw query fetched directly via ChromaDB
  metadata filter. Deterministic, no API call, no reranking.

  Step 2 — dense search (widened to config.RERANK_POOL_K) and rewrite
  (ALSO widened to RERANK_POOL_K) both run unconditionally, no gate.
  Unchanged from the prior revision.

  Step 3 — MERGE, REVISED. The prior revision flattened dense + rewrite
  into one deduped pool and reranked it ONCE against the raw query.
  Diagnosed 2026-07-21 (q015, q010, q003): that flat approach silently
  discarded a protection retrieve_with_rewrite()'s OWN internal merge
  always had — guaranteeing each sub-query's best result a seat regardless
  of how it scores against the compound raw query. Restored here, ported
  from distance-based to rerank-score-based:

    GUARANTEED SLOT — each sub-query's own best-reranked candidate, scored
    against ITS OWN sub-query text (not the raw query), gets a seat first.
    This is what a category-suppression failure like q015's needs: T1190's
    chunk scored terribly against a CVE-dominated compound query, but
    would score normally against its own MITRE-phrased sub-query.

    FILL — remaining slots, from whatever's left, reranked against the
    ORIGINAL raw query — one shared, comparable scoring surface.

  Cross-sub-query score comparability is explicitly NOT assumed for
  guaranteed slots (a cross-encoder score is calibrated per query text,
  same caution retrieve_with_rewrite() already documents for raw
  distances). Only the FILL step's scores are mutually comparable.

  ALSO NEW — header stripping (see reranker.py): confirmed on q010 that
  ingest.py's verbatim-embedded TACTIC:/TECHNIQUE_ID: header lines were
  winning cross-encoder matches on pure header-phrase overlap, unrelated
  to chunk content. Fixed inside rerank() itself — applies everywhere
  automatically, no separate wiring needed here.

  STILL REMOVED: the confidence gate (is_confident(), threshold,
  calibration). No threshold anywhere in this file.

  STILL OPEN, NOT ADDRESSED BY THIS REVISION: a statistical near-tie just
  below the top-k cutoff (confirmed on q003 — three candidates within a
  0.02 score spread, one slot too shallow) isn't fixed by guaranteed slots
  or header-stripping. The guaranteed-slot restoration MAY help it as a
  side effect (rewrite's canonical MITRE phrasing could score better than
  the raw query did) — unconfirmed until re-tested, not claimed as fixed.

  KNOWN OPEN RISK, UNCHANGED FROM PRIOR REVISION: dedup key is still
  technique_id. T1140-style chunk fragmentation (Known Limitation #1) is
  now confirmed to affect at least 5 techniques (T1140, T1027, T1202,
  T1650, T1659 — found incidentally across diagnostic runs, not a targeted
  sweep), not just T1140. Not fixed here.
"""

import logging

from exact_id import extract_exact_ids
from query_rewrite import retrieve_with_rewrite
from reranker import rerank

import config

logger = logging.getLogger(__name__)


def _guaranteed_slot_rerank_merge(
    query: str,
    dense_candidates: list[tuple[dict, str]],
    subquery_pools: list[dict],
    k: int,
) -> list[tuple[dict, str, float]]:
    """
    Replaces the flat dedupe-then-single-rerank approach from the prior
    revision. See module docstring for the full rationale.

    dense_candidates: raw (metadata, document_text) pairs from dense
    search on the RAW query — not yet deduped, not yet scored.

    subquery_pools: [{"sub_query_text": str, "candidates": [(meta, doc),
    ...]}, ...] from retrieve_with_rewrite(return_subquery_pools=True) —
    each sub-query's OWN raw candidates, before that function's internal
    merge would otherwise collapse them.

    Returns up to k (metadata, document_text, score) tuples. Order:
    guaranteed picks first (in subquery_pools order), then fill picks by
    descending fill-score — same non-strict-global-ordering behavior
    retrieve_with_rewrite()'s own original merge already has; not a new
    convention invented here.
    """
    seen: set[str] = set()
    final: list[tuple[dict, str, float]] = []

    # ── Guaranteed slots — one per sub-query, scored against ITS OWN text ──
    for sq_pool in subquery_pools:
        sq_text = sq_pool["sub_query_text"]
        sq_candidates = sq_pool["candidates"]
        if not sq_candidates:
            continue
        sq_ranked = rerank(sq_text, sq_candidates, top_k=len(sq_candidates))
        for meta, doc, score in sq_ranked:
            tid = meta.get("technique_id")
            if tid not in seen:
                seen.add(tid)
                final.append((meta, doc, score))
                break  # one guaranteed slot per sub-query

    # ── Fill remaining slots — one shared pool (dense + leftover sub-query
    #    candidates), scored against the ORIGINAL raw query so these scores
    #    ARE mutually comparable (unlike the guaranteed slots above). ──
    remaining: list[tuple[dict, str]] = []
    remaining_seen: set[str] = set()
    for meta, doc in dense_candidates:
        tid = meta.get("technique_id")
        if tid not in seen and tid not in remaining_seen:
            remaining_seen.add(tid)
            remaining.append((meta, doc))
    for sq_pool in subquery_pools:
        for meta, doc in sq_pool["candidates"]:
            tid = meta.get("technique_id")
            if tid not in seen and tid not in remaining_seen:
                remaining_seen.add(tid)
                remaining.append((meta, doc))

    if remaining:
        fill_ranked = rerank(query, remaining, top_k=len(remaining))
        for meta, doc, score in fill_ranked:
            if len(final) >= k:
                break
            tid = meta.get("technique_id")
            if tid not in seen:
                seen.add(tid)
                final.append((meta, doc, score))

    return final[:k]


def retrieve_for_route(
    db,
    query: str,
    client,
    k: int,
    corpus: str | None,
    throttle_fn=None,
    return_pool: bool = False,
) -> dict:
    """
    corpus="mitre" / "kev" -> single-corpus route.
    corpus=None -> BOTH route; sub-queries corpus-routed individually
                   inside the rewrite step, and the dense-search call
                   below also searches the whole store unfiltered.

    Step 1 — exact match: unchanged. Returns immediately if any literal ID
    matched storage. path="exact_match".

    Step 2/3 — dense (widened) + rewrite (widened, per-sub-query pools
    exposed) → guaranteed-slot merge → rerank. path="merged_reranked". See
    module docstring for the full mechanism and why it changed.

    return_pool: if True, includes the FULL union of raw candidates
    (deduped, PRE-final-selection) under "candidate_pool" — for the
    coverage-vs-ranking diagnostic.

    Returns the standard chroma-shaped dict (documents/metadatas/distances/
    error) PLUS "path": "exact_match" | "merged_reranked".

    *** "distances" SEMANTICS: cross-encoder score on the merged_reranked
    path (HIGHER = more relevant) — opposite of cosine distance. Guaranteed
    slots and fill slots come from DIFFERENT scoring surfaces (see module
    docstring) — this field is not even internally consistent within one
    row's own top-k, let alone comparable across rows. Treat it as "this
    candidate's own relevance signal," not as a sortable/comparable score
    across the row. ***

    Raises RuntimeError if a non-exact-match query's total unique candidate
    count (dense ∪ every sub-query pool, deduped) comes up short of k.
    """
    exact_ids = extract_exact_ids(query)

    if exact_ids:
        documents, metadatas = [], []
        for exact_id, exact_corpus in exact_ids:
            result = db.collection.get(
                where={
                    "$and": [
                        {"corpus": {"$eq": exact_corpus}},
                        {"technique_id": {"$eq": exact_id}},
                    ]
                },
                include=["documents", "metadatas"],
            )
            documents.extend(result["documents"])
            metadatas.extend(result["metadatas"])

        if documents:
            n = min(k, len(documents))
            out = {
                "documents": [documents[:n]],
                "metadatas": [metadatas[:n]],
                "distances": [[0.0] * n],
                "error": None,
                "path": "exact_match",
            }
            if return_pool:
                out["candidate_pool"] = list(zip(metadatas[:n], documents[:n]))
            return out
        # IDs present in the text but matched nothing in storage — fall
        # through to dense+rewrite rather than returning empty.

    pool_k = config.RERANK_POOL_K

    dense_result = db.semantic_search(query, n_results=pool_k, corpus=corpus)
    dense_candidates: list[tuple[dict, str]] = []
    if dense_result.get("error"):
        logger.warning(
            f"Dense search failed for '{query[:50]}...': {dense_result['error']} "
            f"— continuing with rewrite pool only."
        )
    else:
        dense_candidates = list(zip(dense_result["metadatas"][0], dense_result["documents"][0]))

    rewrite_result = retrieve_with_rewrite(
        db, query, client, n_results=pool_k, corpus=corpus,
        throttle_fn=throttle_fn, return_subquery_pools=True,
    )
    if rewrite_result.get("error"):
        logger.warning(
            f"Rewrite retrieval failed for '{query[:50]}...': {rewrite_result['error']} "
            f"— continuing with dense pool only."
        )
    subquery_pools = rewrite_result.get("subquery_pools", [])

    # Fail-loud pool-size guard: unique technique_id count across the full
    # union (dense ∪ every sub-query pool), before any final-selection
    # logic runs.
    all_ids: set[str] = {m.get("technique_id") for m, _ in dense_candidates}
    for sq_pool in subquery_pools:
        all_ids |= {m.get("technique_id") for m, _ in sq_pool["candidates"]}
    if len(all_ids) < k:
        raise RuntimeError(
            f"Total unique candidate pool for '{query[:50]}...' has only "
            f"{len(all_ids)} unique technique_id(s), fewer than k={k}. Both "
            f"dense (widened to {pool_k}) and rewrite (widened to {pool_k}, "
            f"{len(subquery_pools)} sub-quer{'y' if len(subquery_pools) == 1 else 'ies'}) "
            f"were searched — a shortfall this large suggests an upstream "
            f"problem, not a legitimately sparse result set."
        )

    top = _guaranteed_slot_rerank_merge(query, dense_candidates, subquery_pools, k)

    out = {
        "documents": [[doc for _, doc, _ in top]],
        "metadatas": [[meta for meta, _, _ in top]],
        "distances": [[score for _, _, score in top]],
        "error": None,
        "path": "merged_reranked",
    }
    if return_pool:
        pool: list[tuple[dict, str]] = []
        pool_seen: set[str] = set()
        for meta, doc in dense_candidates:
            tid = meta.get("technique_id")
            if tid not in pool_seen:
                pool_seen.add(tid)
                pool.append((meta, doc))
        for sq_pool in subquery_pools:
            for meta, doc in sq_pool["candidates"]:
                tid = meta.get("technique_id")
                if tid not in pool_seen:
                    pool_seen.add(tid)
                    pool.append((meta, doc))
        out["candidate_pool"] = pool
    return out