from exact_id import extract_exact_ids
from query_rewrite import retrieve_with_rewrite
from confidence_gate import is_confident, GateNotCalibratedError

import config


def retrieve_for_route(
    db,
    query: str,
    client,
    k: int,
    corpus: str | None,
    throttle_fn=None,
    use_confidence_gate: bool = False,
) -> dict:
    """
    corpus="mitre" / "kev" -> single-corpus route.
    corpus=None -> BOTH route, sub-queries corpus-routed individually
                   inside the rewrite step. The dense-search-first check
                   below (Step 2) searches the whole store unfiltered on
                   this route, same as blind retrieval, since there's no
                   single corpus to narrow to before a split happens.

    Step 1 — exact match: collects ALL literal CVE/technique IDs in the
    raw query (any count, any corpus mix), fetches each directly. Local
    ChromaDB metadata lookup — never throttled, no API call.

    Step 2 — dense search, gated (only runs if use_confidence_gate=True):
    a single un-split semantic_search on the raw query. If the top-1
    result's distance is at or below the calibrated confidence threshold
    (config.RETRIEVAL_CONFIDENCE_THRESHOLD), trust it and return directly
    — no LLM call, no exposure to the rewrite step's known temperature=0
    non-determinism. This is what targets the q001/q004 regressions:
    those queries were already correct on dense search alone, and only
    broke when rewrite ran on them unconditionally.

    Step 3 — rewrite fallback: runs when no literal IDs were found, IDs
    matched nothing in storage, OR the gate isn't confident (or is off,
    reproducing today's always-rewrite behavior exactly). Makes a live
    Gemini call — throttle_fn (if provided) is invoked immediately
    before it, so a caller sharing one throttle across routing + rewrite
    calls (e.g. the eval harness) enforces a single consistent rate-limit
    gap rather than each call site throttling independently.

    Returns the usual chroma-shaped dict (documents/metadatas/distances/
    error), plus "path": "exact_match" | "dense_confident" | "rewrite" —
    read by eval_retrieval.py to report how often each branch fires.
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
            return {
                "documents": [documents[:n]],
                "metadatas": [metadatas[:n]],
                "distances": [[0.0] * n],
                "error": None,
                "path": "exact_match",
            }
        # IDs were present in the text but matched nothing in storage —
        # fall through to dense/rewrite rather than returning empty.

    if use_confidence_gate:
        dense_result = db.semantic_search(query, n_results=k, corpus=corpus)

        if not dense_result.get("error"):
            try:
                confident = is_confident(
                    dense_result, config.RETRIEVAL_CONFIDENCE_THRESHOLD
                )
            except GateNotCalibratedError:
                # Fail loud, but don't crash retrieval over it — a missing
                # calibration is a config problem to fix, not a reason to
                # take down every query. Log it and fall through to the
                # always-safe rewrite path, same as gate=False behavior.
                import logging
                logging.getLogger(__name__).error(
                    "use_confidence_gate=True but "
                    "config.RETRIEVAL_CONFIDENCE_THRESHOLD is unset — "
                    "run calibrate_threshold() first. Falling through to "
                    "rewrite for this query."
                )
                confident = False

            if confident:
                dense_result["path"] = "dense_confident"
                return dense_result
        # Not confident, dense search errored, or gate wasn't calibrated —
        # fall through to rewrite exactly as the ungated path always has.

    rewrite_result = retrieve_with_rewrite(
        db, query, client, n_results=k, corpus=corpus, throttle_fn=throttle_fn
    )
    rewrite_result["path"] = "rewrite"
    return rewrite_result