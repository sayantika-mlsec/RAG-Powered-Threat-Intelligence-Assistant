"""
Confidence gate for the fallback-only rewrite restructuring.

Works directly on the chroma-shaped result dict every retrieval call in
this project already returns ({"documents": [[...]], "metadatas": [[...]],
"distances": [[...]], "error": ...}) -- no wrapper dataclass, since one
would only be translating a shape that's already consistent everywhere
else in retrieval_pipeline.py.

Why a threshold, and why calibrated offline: see eval_retrieval.py /
retrieval_pipeline.py docstrings. Short version -- there's no ground
truth at inference time, so we can't check "did I retrieve the right
chunk." We CAN check the chunk's distance (how close its embedding is
to the query's), and that distance is correlated with correctness. Not
proof of correctness, just the same tell a librarian's "this is almost
exactly what you asked for" vs "closest thing on the shelf" gives you.
Calibrate that correlation once, offline, against eval_set.json, where
correctness IS known.
"""

import logging
from typing import Callable, Optional


class GateNotCalibratedError(RuntimeError):
    """Raised if the gate is used with no threshold set.

    Fail loud: a silent default threshold would quietly change
    retrieval behavior with no record of why -- same discipline as
    'resp.parsed is None raises, not defaults to both' elsewhere in
    this project.
    """
    pass


def top1_distance(chroma_result: dict) -> Optional[float]:
    """Best (smallest) distance in a chroma-shaped result, or None if
    there were zero hits."""
    distances = chroma_result.get("distances")
    if not distances or not distances[0]:
        return None
    return distances[0][0]


def is_confident(chroma_result: dict, threshold: Optional[float]) -> bool:
    """True if dense search alone is trusted for this query.

    threshold is passed in explicitly (not read from config here) so
    this function has no hidden import-time dependency -- callers own
    getting the calibrated value from config.py.
    """
    if threshold is None:
        raise GateNotCalibratedError(
            "No confidence threshold provided. Run calibrate_threshold() "
            "against eval_set.json, set config.RETRIEVAL_CONFIDENCE_THRESHOLD "
            "to the result, and pass that value in."
        )
    d = top1_distance(chroma_result)
    if d is None:
        return False
    return d <= threshold


def calibrate_threshold(
    eval_queries: list[dict],
    dense_search_fn: Callable[[str, Optional[str]], dict],
    percentile: float = 75.0,
) -> dict:
    """Run dense-search-only over the eval set and pick a threshold.

    eval_queries: rows shaped like {"query": str, "corpus": str | None,
                  "expected_ids": list[str]} -- pull straight from
                  eval_set.json (expected_technique_ids | expected_cve_ids
                  merged into expected_ids the same way eval_retrieval.py
                  already does).
    dense_search_fn: e.g. lambda q, c: db.semantic_search(q, n_results=3,
                      corpus=c) -- same call retrieve_for_route() makes.

    NOTE on the "both" route (corpus=None): a blind whole-store search
    can score a deceptively good top-1 distance on one half of a
    genuinely two-part query and miss that splitting was ever needed.
    Worth checking the success/failure separation for corpus=None rows
    specifically, not just in aggregate, before trusting one threshold
    across all three route types. Don't build a per-route-type threshold
    unless the aggregate one actually shows this problem -- check first.

    Returns the recommended threshold AND the raw distributions, so you
    look at the separation yourself rather than trusting an auto-picked
    number blindly -- this is a 30-50 query eval set, not enough data
    to trust a percentile without a look.
    """
    success_distances = []
    failure_distances = []

    for row in eval_queries:
        result = dense_search_fn(row["query"], row.get("corpus"))
        d = top1_distance(result)
        if d is None:
            failure_distances.append(float("inf"))
            continue

        top_metadata = result["metadatas"][0][0]
        expected = set(row["expected_ids"])
        if top_metadata.get("technique_id") in expected:
            success_distances.append(d)
        else:
            failure_distances.append(d)

    success_distances.sort()
    failure_distances.sort()

    if not success_distances:
        raise RuntimeError(
            "Zero successful dense-only queries in the eval set -- "
            "calibration is meaningless here. Check dense_search_fn "
            "before trusting any threshold."
        )

    # percentile-based: "cover this % of queries dense search already
    # gets right, on their own distance terms." Coverage-optimized, but
    # blind to how many failures fall below the same cutoff -- on
    # overlapping distributions it can trust genuinely wrong results.
    idx = min(
        int(len(success_distances) * percentile / 100),
        len(success_distances) - 1,
    )
    recommended = success_distances[idx]

    # conservative: the largest success distance still below the
    # smallest observed failure distance. Zero observed false-"confident"
    # calls -- every failure in this eval set stays excluded. This is
    # the one to actually use: a wrongly-trusted dense result is
    # guaranteed wrong with no rewrite chance to correct it, while a
    # success sent to rewrite unnecessarily is only sometimes wrong
    # (rewrite mostly holds -- see the Jul 13 findings). The two error
    # types are not equal cost, so "optimize coverage" is the wrong
    # objective; "never trust a failure" is the right one.
    if failure_distances:
        min_failure = failure_distances[0]
        safe_successes = [d for d in success_distances if d < min_failure]
        conservative = safe_successes[-1] if safe_successes else None
    else:
        # No failures at all in this eval set -- nothing to guard
        # against; trust dense search up to its worst observed success.
        conservative = success_distances[-1]

    if conservative is None:
        logging.getLogger(__name__).warning(
            "No conservative threshold exists -- every success distance "
            "is >= the smallest failure distance. The two distributions "
            "don't separate at all; a single top-1-distance signal isn't "
            "viable here. Don't ship any threshold from this run -- this "
            "is the 'insufficient, needs margin-based signal' case."
        )

    return {
        "recommended_threshold": recommended,
        "conservative_threshold": conservative,
        "success_distances": success_distances,
        "failure_distances": failure_distances,
        "n_success": len(success_distances),
        "n_failure": len(failure_distances),
    }