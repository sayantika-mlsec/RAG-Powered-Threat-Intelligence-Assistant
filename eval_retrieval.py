import json
import logging
import mlflow
import math
import tempfile
import os
from collections import defaultdict
from ingest import ThreatIntelDB

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
# faithfulness gate expects if K matches this pin. If config.RETRIEVAL_TOP_K ever
# diverges from 3, eligibility semantics silently change — so we guard, not assume.
# Real if/raise, NOT assert (assert is stripped under `python -O`).
PINNED_BASELINE_K = 3

# ─── Evaluation Pipeline ──────────────────────────────────────────────────────

def run_evaluation(dataset_path: str, K: int = config.RETRIEVAL_TOP_K):
    """
    Evaluates the RAG pipeline using Precision@K and Recall@K, and emits a
    per-row metrics artifact (recall@K + eligibility) to the active MLflow run.

    Requires ingest.py to be present in the same directory.

    Returns:
        dict {"precision_overall": float, "recall_overall": float} on a completed
        run, or None if evaluation did not run (dataset load failure or no
        eligible queries). Callers verifying against a pin must handle None.

    Fail-loud invariants (artifact-contract, not warnings — a breach raises
    BEFORE any artifact is written, so a broken run never ships a clean file):
      - K must equal the pinned baseline K, else eligibility semantics drift.
      - Corpus stamp must be readable, else the artifact's lineage is blank.
      - Per-row sums must reconcile to the global aggregate.
    """
    # ── Guard: K must match the pinned baseline ──
    if K != PINNED_BASELINE_K:
        raise ValueError(
            f"K={K} does not match the pinned baseline K={PINNED_BASELINE_K}. "
            f"The per-row 'eligible' flags and recall@K would mean something "
            f"different from the baseline the faithfulness gate keys against. "
            f"Re-pin the baseline deliberately or run at K={PINNED_BASELINE_K}."
        )

    # 1. Load the ground-truth dataset
    try:
        with open(dataset_path, 'r', encoding='utf-8') as f:
            all_queries = json.load(f)
    except Exception as e:
        logger.error(f"Failed to load dataset: {e}")
        return None

    # 2. Filter for relevant categories (Group A: at least one expected ID populated)
    group_a_queries = [
        q for q in all_queries
        if len(q.get('expected_technique_ids', [])) > 0 or len(q.get('expected_cve_ids', [])) > 0
    ]

    if not group_a_queries:
        logger.warning("No valid queries found for evaluation.")
        return None

    # 3. Initialize ChromaDB Connection
    db = ThreatIntelDB()

    # ─── Corpus Stamp Extraction ───
    # name is a CONSTANT label ("mitre_threat_intel", hardcoded in the wrapper);
    # count is the live lineage signal. Accessors confirmed against ingest.py.
    # Guard stays as cheap insurance — if the wrapper changes, fail loud with a
    # clear message rather than mis-stamp. Known limit recorded in the payload's
    # corpus_stamp.note: a count stamp cannot detect content-only edits (e.g. the
    # TACTIC fix changed chunk text, not chunk count). Corpus cleanliness is
    # guaranteed by run lineage, not by this stamp.
    try:
        corpus_name = db.collection.name
        corpus_size = db.collection.count()
    except AttributeError as e:
        raise RuntimeError(
            "Could not read ChromaDB collection state for the corpus stamp "
            "(expected db.collection.name / db.collection.count()). Fix the "
            "wrapper accessor before proceeding."
        ) from e

    metrics_tracker = defaultdict(lambda: {"precision_sum": 0.0, "recall_sum": 0.0, "count": 0})

    # ─── Per-Row Tracking ───
    # INVARIANT: every append to per_row_records below sits under the SAME guards
    # as the global_recall_sum increment. Reconciliation only holds because of
    # this coupling. Do NOT add a `continue` between the append and the global
    # sum, or reconciliation will silently lie.
    per_row_records = []

    logger.info(f"Starting evaluation of {len(group_a_queries)} queries at K={K}...")

    # ── GLOBAL HEADLINE TRACKERS ──
    global_precision_sum = 0.0
    global_recall_sum = 0.0
    global_count = 0

    mlflow.log_metric("eval_dataset_size", len(group_a_queries))

    # 4. Core Evaluation Loop
    for query_row in group_a_queries:

        expected_ids = set(query_row['expected_technique_ids']) | set(query_row['expected_cve_ids'])

        if not expected_ids:
            continue

        results = db.semantic_search(query_row['query'], n_results=K)

        if results.get("error"):
            logger.error(f"Search failed for query: {query_row['query'][:40]}... Error: {results['error']}")
            continue

        retrieved_ids = [m['technique_id'] for m in results['metadatas'][0]]

        # Original console outputs untouched
        print(f"Query: {query_row['query']}")
        print(f"Expected: {expected_ids}")
        print(f"Retrieved: {retrieved_ids}")

        # ── Metric Calculation ──
        relevant_retrieved_count = len(set(retrieved_ids) & expected_ids)

        precision = relevant_retrieved_count / len(retrieved_ids) if retrieved_ids else 0.0
        recall = relevant_retrieved_count / len(expected_ids)

        # ── Record Per-Row Data ──
        # (append + global sum below are under identical guards — see INVARIANT)
        row_id = query_row.get('id')
        if not row_id:
            logger.warning(
                f"Row missing 'id' — downstream faithfulness join will break for "
                f"this row: '{query_row['query'][:40]}...'"
            )

        per_row_records.append({
            "id": row_id,
            f"recall_at_{K}": recall,
            "eligible": recall > 0
        })

        # ── Aggregation: Granular ──
        cat = query_row.get('category', 'unknown')
        diff = query_row.get('difficulty', 'unknown')
        group_key = (cat, diff)

        metrics_tracker[group_key]["precision_sum"] += precision
        metrics_tracker[group_key]["recall_sum"] += recall
        metrics_tracker[group_key]["count"] += 1

        # ── Aggregation: Global ──
        global_precision_sum += precision
        global_recall_sum += recall
        global_count += 1

    precision_overall = global_precision_sum / global_count if global_count > 0 else 0.0
    recall_overall = global_recall_sum / global_count if global_count > 0 else 0.0

    # 5. Output Summary Report
    print(f"\n--- Retrieval Evaluation Summary (K={K}) ---")
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

    # ─── Reconciliation (fail-loud) & Artifact Generation ───

    # 1. Reconcile per-row sum against the global aggregate. These sum the same
    #    values in the same order, so they should be bit-identical — a tight
    #    tolerance catches real divergence instead of masking it. Verified, not
    #    assumed: a breach raises BEFORE the artifact is written.
    per_row_recall_sum = sum(r[f"recall_at_{K}"] for r in per_row_records)
    if not math.isclose(per_row_recall_sum, global_recall_sum, rel_tol=1e-9, abs_tol=1e-12):
        raise AssertionError(
            f"Reconciliation FAILED: per-row recall sum ({per_row_recall_sum}) "
            f"!= global recall sum ({global_recall_sum}). The per-row artifact "
            f"disagrees with the logged aggregate; refusing to write it."
        )
    logger.info("Reconciliation OK: per-row recall sum matches global aggregate.")

    # 2. Construct Payload
    artifact_payload = {
        "corpus_stamp": {
            "collection_name": corpus_name,
            "chunk_count": corpus_size,
            "note": (
                "Tracks chunk count only. Content-only edits at constant count "
                "(e.g. the TACTIC fix) are NOT detected; corpus version is "
                "guaranteed by run lineage, not this stamp."
            ),
        },
        "k_value": K,
        "rows": per_row_records,
    }

    # 3. Dump to a temp file & log to MLflow, cleaning up on every path.
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as tmp:
            json.dump(artifact_payload, tmp, indent=2)
            tmp_path = tmp.name
        mlflow.log_artifact(tmp_path, "per_row_metrics")
        logger.info("Per-row metrics artifact written to MLflow.")
    finally:
        # Runs even if log_artifact raises — no orphaned temp file.
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)

    # ── Return overall metrics for caller-side baseline verification ──
    # Single source: the values just computed, not re-read from MLflow.
    return {"precision_overall": precision_overall, "recall_overall": recall_overall}


if __name__ == "__main__":
    run_evaluation(str(config.EVAL_SET_PATH))