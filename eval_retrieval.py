import json
import logging
import mlflow
import math
import tempfile
import os
import time
from collections import defaultdict
from pathlib import Path
from dotenv import load_dotenv

# Load .env BEFORE importing config. config.py reads GCP_PROJECT_ID /
# GCP_LOCATION at import time (module-level), so if load_dotenv only ran
# later — conditionally, inside __main__, only for the routed arm, as it did
# before this migration — plain `import config` could fail before a single
# line of this script's logic runs, even for the blind arm that needs no
# Gemini calls at all. Loading here, unconditionally, first, removes that
# hidden ordering dependency entirely.
load_dotenv(dotenv_path=Path(__file__).parent / ".env", override=True)

from ingest import ThreatIntelDB
from routing import route_query, Route

import config

# ─── Logging Setup ────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# ─── Pinned Baseline ──────────────────────────────────────────────────────────
# The retrieval baseline (precision 0.2444 / recall 0.5667) was captured at K=3.
# The per-row `eligible` flag and recall@K only mean what the downstream
# faithfulness gate expects if K matches this pin. Real if/raise, NOT assert.
PINNED_BASELINE_K = 3

# Throttle between routing API calls (routed arm only). Blind arm makes NO API
# calls — semantic_search is local ChromaDB — so it never throttles.
THROTTLE_SECONDS = 6


def _route_to_corpus(route: Route) -> str | None:
    """Map a Route to the `corpus` arg for semantic_search. SKIP is handled by
    the caller (no retrieval), so it is not mapped here — an unmapped route
    raises rather than silently defaulting to whole-store (which would corrupt
    the A/B by reintroducing blind retrieval)."""
    if route == Route.MITRE_ONLY:
        return "mitre"
    if route == Route.KEV_ONLY:
        return "kev"
    if route == Route.BOTH:
        return None
    raise ValueError(f"Unmapped non-skip route: {route!r}")


# ─── Evaluation Pipeline ──────────────────────────────────────────────────────

def run_evaluation(
    dataset_path: str,
    K: int = config.RETRIEVAL_TOP_K,
    *,
    use_routing: bool = False,
    router_client=None,
):
    """
    Evaluate the RAG retrieval with Precision@K and Recall@K, emit a per-row
    metrics artifact (recall@K + eligibility + route) to the active MLflow run.

    use_routing is the ONLY A/B variable:
        False -> blind baseline: whole-store semantic_search (reproduces
                 precision 0.2444 / recall 0.5667). No API calls, no throttle.
        True  -> agentic: route_query decides the corpus, retrieval is filtered
                 to it. A scored row wrongly routed to skip retrieves NOTHING
                 (recall 0) — the misroute is penalised, not hidden.

    Gating (route-based, three-way — restraint vs misroute vs normal):
        - gated-out = rows with NO expected IDs (nothing to retrieve; correct
                      behavior is no-retrieval). Never scored, either arm.
        - scored    = rows WITH expected IDs. A skip-route here is a misroute →
                      empty retrieval → recall 0.
        Reconciliation invariant: len(scored) + len(gated_out) == total.

    Returns {"precision_overall": float, "recall_overall": float} on a completed
    run, or None if evaluation did not run.
    """
    # ── Guard: K must match the pinned baseline ──
    if K != PINNED_BASELINE_K:
        raise ValueError(
            f"K={K} does not match the pinned baseline K={PINNED_BASELINE_K}. "
            f"The per-row 'eligible' flags and recall@K would mean something "
            f"different from the baseline the faithfulness gate keys against. "
            f"Re-pin the baseline deliberately or run at K={PINNED_BASELINE_K}."
        )

    if use_routing and router_client is None:
        raise ValueError("use_routing=True requires a router_client.")

    # 1. Load the ground-truth dataset
    try:
        with open(dataset_path, 'r', encoding='utf-8') as f:
            all_queries = json.load(f)
    except Exception as e:
        logger.error(f"Failed to load dataset: {e}")
        return None

    total = len(all_queries)

    # 2. Partition: scored (has expected IDs) vs gated-out (nothing to retrieve).
    #    Route-based misrouting is measured WITHIN scored, not by moving rows
    #    between these two groups — so reconciliation stays keyed off the data.
    def _has_expected(q):
        return len(q.get('expected_technique_ids', [])) > 0 or len(q.get('expected_cve_ids', [])) > 0

    scored_queries = [q for q in all_queries if _has_expected(q)]
    gated_out      = [q for q in all_queries if not _has_expected(q)]

    # Reconcile the partition BEFORE any work — a mismatch means a row fell
    # through both predicates (malformed row) and the denominator is wrong.
    if len(scored_queries) + len(gated_out) != total:
        raise RuntimeError(
            f"Partition failed: {len(scored_queries)} scored + {len(gated_out)} "
            f"gated-out != {total} total."
        )

    if not scored_queries:
        logger.warning("No scored queries (all rows gated out). Nothing to evaluate.")
        return None

    # 3. Initialize ChromaDB Connection
    db = ThreatIntelDB()

    # ─── Corpus Stamp Extraction ───
    try:
        corpus_name = db.collection.name
        corpus_size = db.collection.count()
    except AttributeError as e:
        raise RuntimeError(
            "Could not read ChromaDB collection state for the corpus stamp "
            "(expected db.collection.name / db.collection.count())."
        ) from e

    metrics_tracker = defaultdict(lambda: {"precision_sum": 0.0, "recall_sum": 0.0, "count": 0})
    per_row_records = []

    logger.info(
        f"Evaluating {len(scored_queries)} scored queries at K={K} "
        f"(use_routing={use_routing}, {len(gated_out)} gated out)..."
    )

    global_precision_sum = 0.0
    global_recall_sum = 0.0
    global_count = 0
    misroute_ids = []   # scored rows the router sent to skip — routing failures

    mlflow.log_metric("eval_dataset_size", len(scored_queries))
    mlflow.log_metric("n_gated_out", len(gated_out))
    mlflow.log_metric("eval_total_rows", total)

    made_live_call = False

    # 4. Core Evaluation Loop (scored rows only)
    for query_row in scored_queries:
        expected_ids = set(query_row['expected_technique_ids']) | set(query_row['expected_cve_ids'])

        route_value = None

        if use_routing:
            # Throttle before each routing call except the first.
            if made_live_call:
                time.sleep(THROTTLE_SECONDS)
            decision = route_query(query_row['query'], router_client)
            made_live_call = True
            route_value = decision.route.value

            if decision.route == Route.SKIP:
                # Misroute: a real query (has expected IDs) wrongly skipped.
                # Retrieval is empty → recall 0. Penalised, not hidden.
                retrieved_ids = []
                misroute_ids.append(query_row.get('id'))
                logger.warning(
                    f"MISROUTE {query_row.get('id')}: scored query routed to skip."
                )
            else:
                corpus = _route_to_corpus(decision.route)
                results = db.semantic_search(query_row['query'], n_results=K, corpus=corpus)
                if results.get("error"):
                    logger.error(f"Search failed: {query_row['query'][:40]}... {results['error']}")
                    continue
                retrieved_ids = [m['technique_id'] for m in results['metadatas'][0]]
        else:
            # Blind baseline — whole store, no filter, no API call.
            results = db.semantic_search(query_row['query'], n_results=K)
            if results.get("error"):
                logger.error(f"Search failed: {query_row['query'][:40]}... {results['error']}")
                continue
            retrieved_ids = [m['technique_id'] for m in results['metadatas'][0]]

        # ── Metric Calculation ──
        relevant_retrieved_count = len(set(retrieved_ids) & expected_ids)
        precision = relevant_retrieved_count / len(retrieved_ids) if retrieved_ids else 0.0
        recall = relevant_retrieved_count / len(expected_ids)

        row_id = query_row.get('id')
        if not row_id:
            logger.warning(f"Row missing 'id': '{query_row['query'][:40]}...'")

        per_row_records.append({
            "id": row_id,
            f"recall_at_{K}": recall,
            "eligible": recall > 0,
            "route": route_value,   # None on the blind arm
        })

        cat = query_row.get('category', 'unknown')
        diff = query_row.get('difficulty', 'unknown')
        group_key = (cat, diff)
        metrics_tracker[group_key]["precision_sum"] += precision
        metrics_tracker[group_key]["recall_sum"] += recall
        metrics_tracker[group_key]["count"] += 1

        global_precision_sum += precision
        global_recall_sum += recall
        global_count += 1

    precision_overall = global_precision_sum / global_count if global_count > 0 else 0.0
    recall_overall = global_recall_sum / global_count if global_count > 0 else 0.0

    # 5. Summary Report
    print(f"\n--- Retrieval Evaluation Summary (K={K}, use_routing={use_routing}) ---")
    print(f"{'Category':<15} | {'Difficulty':<10} | {'Precision@K':<12} | {'Recall@K'}")
    print("-" * 55)
    for (cat, diff), metrics in sorted(metrics_tracker.items()):
        avg_precision = metrics["precision_sum"] / metrics["count"]
        avg_recall = metrics["recall_sum"] / metrics["count"]
        print(f"{cat:<15} | {diff:<10} | {avg_precision:<12.4f} | {avg_recall:.4f}")
        mlflow.log_metric(f"precision_{cat}_{diff}", avg_precision)
        mlflow.log_metric(f"recall_{cat}_{diff}", avg_recall)
    print("-" * 55)

    mlflow.log_metric("precision_overall", precision_overall)
    mlflow.log_metric("recall_overall", recall_overall)
    if use_routing:
        mlflow.log_metric("n_misroutes", len(misroute_ids))

    # ─── Reconciliation (fail-loud) & Artifact ───
    per_row_recall_sum = sum(r[f"recall_at_{K}"] for r in per_row_records)
    if not math.isclose(per_row_recall_sum, global_recall_sum, rel_tol=1e-9, abs_tol=1e-12):
        raise AssertionError(
            f"Reconciliation FAILED: per-row recall sum ({per_row_recall_sum}) "
            f"!= global recall sum ({global_recall_sum})."
        )
    # Row-count reconciliation: scored rows evaluated + gated == total.
    if len(per_row_records) + len(gated_out) != total:
        raise AssertionError(
            f"Row reconciliation FAILED: {len(per_row_records)} scored + "
            f"{len(gated_out)} gated-out != {total} total. A scored row was "
            f"dropped mid-loop (search error?) — refusing to write the artifact."
        )
    logger.info("Reconciliation OK: recall sums match; scored + gated == total.")

    artifact_payload = {
        "use_routing": use_routing,
        "corpus_stamp": {
            "collection_name": corpus_name,
            "chunk_count": corpus_size,
            "note": (
                "Tracks chunk count only. Content-only edits at constant count "
                "are NOT detected; corpus version is guaranteed by run lineage."
            ),
        },
        "k_value": K,
        "n_gated_out": len(gated_out),
        "gated_out_ids": sorted(q.get('id') for q in gated_out),
        "n_misroutes": len(misroute_ids),
        "misroute_ids": sorted(m for m in misroute_ids if m),
        "rows": per_row_records,
    }

    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as tmp:
            json.dump(artifact_payload, tmp, indent=2)
            tmp_path = tmp.name
        mlflow.log_artifact(tmp_path, "per_row_metrics")
        logger.info("Per-row metrics artifact written to MLflow.")
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)

    return {"precision_overall": precision_overall, "recall_overall": recall_overall}


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--use-routing", action="store_true",
                    help="agentic arm. Omit for the blind baseline.")
    args = ap.parse_args()

    client = None
    if args.use_routing:
        # One shared client, Vertex-backed — no API key, no separate SDK
        # config here. Auth and quota now live with gcloud ADC + the GCP
        # project, not a per-file `genai.Client(api_key=...)` call.
        from gemini_client import get_client
        client = get_client()

    mlflow.set_experiment(config.MLFLOW_EXPERIMENT_NAME)
    run_name = "retrieval_eval_routed" if args.use_routing else "retrieval_eval_blind"
    with mlflow.start_run(run_name=run_name):
        mlflow.set_tag("arm", "routed" if args.use_routing else "blind")

        # ── Config params — make the run self-describing / reproducible ──
        mlflow.log_param("use_routing",     args.use_routing)
        mlflow.log_param("k",               config.RETRIEVAL_TOP_K)
        mlflow.log_param("embedding_model", config.EMBEDDING_MODEL)
        mlflow.log_param("chunk_size",      config.CHUNK_SIZE)
        mlflow.log_param("chunk_overlap",   config.CHUNK_OVERLAP)
        mlflow.log_param("max_chunk_chars", config.MAX_CHUNK_CHARS)
        if args.use_routing:
            # Router model lives in routing.py (_ROUTER_MODEL), not config.
            from routing import _ROUTER_MODEL
            mlflow.log_param("router_model", _ROUTER_MODEL)

        run_evaluation(
            str(config.EVAL_SET_PATH),
            use_routing=args.use_routing,
            router_client=client,
        )