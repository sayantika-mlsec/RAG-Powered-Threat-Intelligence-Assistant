"""
calibrate_confidence.py

Run this once, standalone, to produce the confidence_gate threshold.
Not part of eval_retrieval.py's precision/recall run — this needs a
different path: dense-search-only, routed but with no exact-match
short-circuit and no rewrite, since we're isolating exactly the
question the gate has to answer.

Usage:
    python calibrate_confidence.py
"""

import json
import time
import logging
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(dotenv_path=Path(__file__).parent / ".env", override=True)

from ingest import ThreatIntelDB
from routing import route_query, Route
from exact_id import extract_exact_ids
from confidence_gate import calibrate_threshold
from gemini_client import get_client
import config

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

THROTTLE_SECONDS = 6


def _route_to_corpus(route: Route) -> str | None:
    if route == Route.MITRE_ONLY:
        return "mitre"
    if route == Route.KEV_ONLY:
        return "kev"
    if route == Route.BOTH:
        return None
    raise ValueError(f"Unmapped route: {route!r}")


def main():
    with open(config.EVAL_SET_PATH, "r", encoding="utf-8") as f:
        all_queries = json.load(f)

    db = ThreatIntelDB()
    client = get_client()

    eval_queries = []
    n_scored = 0
    n_skipped_exact = 0
    n_misrouted_skip = 0
    made_call = False

    for row in all_queries:
        expected_ids = set(row.get("expected_technique_ids", [])) | set(row.get("expected_cve_ids", []))
        if not expected_ids:
            continue  # same gating eval_retrieval.py uses — nothing to score
        n_scored += 1

        # Rows with a literal ID in the query text resolve via exact-match
        # in production and never reach the gate — including them here
        # would mix in either trivial 0.0-distance hits or a rare
        # storage-miss edge case, neither of which is what the gate
        # actually has to discriminate on. Skip outright.
        if extract_exact_ids(row["query"]):
            n_skipped_exact += 1
            continue

        if made_call:
            time.sleep(THROTTLE_SECONDS)
        decision = route_query(row["query"], client)
        made_call = True

        if decision.route == Route.SKIP:
            # Misroute: a routing failure, not a retrieval-confidence
            # question — nothing to calibrate from an empty retrieval.
            n_misrouted_skip += 1
            continue

        eval_queries.append({
            "query": row["query"],
            "corpus": _route_to_corpus(decision.route),
            "expected_ids": sorted(expected_ids),
        })

    # Reconciliation, same discipline as eval_retrieval.py's
    # scored + gated_out == total check — every scored row must be
    # accounted for by exactly one of: skipped-exact, misrouted, or used.
    if n_skipped_exact + n_misrouted_skip + len(eval_queries) != n_scored:
        raise RuntimeError(
            f"Reconciliation failed: {n_skipped_exact} skipped-exact + "
            f"{n_misrouted_skip} misrouted + {len(eval_queries)} used "
            f"!= {n_scored} scored. A row fell through uncounted."
        )

    logger.info(
        f"Scored: {n_scored} | skipped (exact-match): {n_skipped_exact} | "
        f"skipped (misrouted to SKIP): {n_misrouted_skip} | "
        f"used for calibration: {len(eval_queries)}"
    )

    def dense_search_fn(q, c):
        return db.semantic_search(q, n_results=config.RETRIEVAL_TOP_K, corpus=c)

    result = calibrate_threshold(eval_queries, dense_search_fn, percentile=75.0)

    print(f"Success distances (n={result['n_success']}): "
          f"{[round(d, 4) for d in result['success_distances']]}")
    print(f"Failure distances (n={result['n_failure']}): "
          f"{[round(d, 4) for d in result['failure_distances']]}")
    print(f"\nCoverage-optimized threshold:  {result['recommended_threshold']:.4f} "
          f"(covers more successes, but may trust some failures — check "
          f"whether any failure distance falls below it)")
    if result["conservative_threshold"] is not None:
        print(f"Conservative threshold:       {result['conservative_threshold']:.4f} "
              f"(zero observed failures trusted — use this one)")
    else:
        print(
            "\nNo conservative threshold exists — success and failure "
            "distances don't separate at all on this eval set. Don't ship "
            "either number; the single-signal gate isn't viable here."
        )
        return

    print(
        f"\nSet in config.py: "
        f"RETRIEVAL_CONFIDENCE_THRESHOLD = {result['conservative_threshold']:.4f}"
    )
    print(
        f"(n={result['n_success'] + result['n_failure']} — small sample. "
        f"'Zero observed false positives' is against this eval set only, "
        f"not a guarantee. Re-check if/when the eval set grows.)"
    )


if __name__ == "__main__":
    main()