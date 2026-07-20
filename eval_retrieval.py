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

# Load .env BEFORE importing config — see original note: config.py reads
# GCP_PROJECT_ID / GCP_LOCATION at import time, so this must run first,
# unconditionally, for every arm (including the blind arm, which makes no
# Gemini calls but still imports config).
load_dotenv(dotenv_path=Path(__file__).parent / ".env", override=True)

from ingest import ThreatIntelDB
from routing import route_query, Route
from retrieval_pipeline import retrieve_for_route

import config

# ─── Logging Setup ────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# ─── Pinned Baseline ──────────────────────────────────────────────────────────
PINNED_BASELINE_K = 3

# Throttle between ANY live Gemini call — routing AND rewrite calls both
# share this. Blind arm makes NO API calls — semantic_search is local
# ChromaDB — so it never throttles. NOTE: post-gate-removal, EVERY
# non-exact-match query in the fixes arm now makes a rewrite call
# (previously the gate skipped rewrite ~3/13 times) — throttling matters
# slightly more than before, not less.
THROTTLE_SECONDS = 6

CORPUS_TAGS = ("mitre", "kev")


def _route_to_corpus(route: Route) -> str | None:
    """Map a Route to the `corpus` arg for semantic_search. SKIP is handled by
    the caller (no retrieval), so it is not mapped here — an unmapped route
    raises rather than silently defaulting to whole-store."""
    if route == Route.MITRE_ONLY:
        return "mitre"
    if route == Route.KEV_ONLY:
        return "kev"
    if route == Route.BOTH:
        return None
    raise ValueError(f"Unmapped non-skip route: {route!r}")


def _corpus_stamp(db: ThreatIntelDB) -> dict:
    """
    Read the collection's chunk counts for the corpus stamp — PER CORPUS,
    reconciled against the combined total. Unchanged by this session's work.
    """
    try:
        collection_name = db.collection.name
        total_count = db.collection.count()
        per_corpus_counts = {
            tag: len(db.collection.get(where={"corpus": {"$eq": tag}}, include=[])["ids"])
            for tag in CORPUS_TAGS
        }
    except AttributeError as e:
        raise RuntimeError(
            "Could not read ChromaDB collection state for the corpus stamp "
            "(expected db.collection.name / db.collection.count() / "
            "db.collection.get())."
        ) from e

    summed = sum(per_corpus_counts.values())
    if summed != total_count:
        raise RuntimeError(
            f"Corpus stamp reconciliation FAILED: "
            f"{' + '.join(f'{tag} ({n})' for tag, n in per_corpus_counts.items())} "
            f"= {summed}, but collection total is {total_count}. A chunk is "
            f"untagged, mistagged, or tagged with a corpus value outside "
            f"{CORPUS_TAGS} — refusing to write a stamp that can't account "
            f"for every chunk."
        )

    return {
        "collection_name": collection_name,
        "chunk_count_total": total_count,
        **{f"chunk_count_{tag}": n for tag, n in per_corpus_counts.items()},
    }


# ─── Evaluation Pipeline ──────────────────────────────────────────────────────

def run_evaluation(
    dataset_path: str,
    K: int = config.RETRIEVAL_TOP_K,
    *,
    use_routing: bool = False,
    use_retrieval_fixes: bool = False,
    router_client=None,
):
    """
    Evaluate the RAG retrieval with Precision@K and Recall@K, emit a per-row
    metrics artifact (recall@K + eligibility + route) to the active MLflow run.

    TWO AXES (was three — the confidence-gate axis is retired, see
    retrieval_pipeline.py module docstring):

        use_routing         -> corpus selection. False = blind whole-store
                                semantic_search (reproduces precision 0.2444 /
                                recall 0.5667). True = agentic route_query
                                decides the corpus, retrieval filtered to it.

        use_retrieval_fixes -> exact-match ID lookup + widened dense search
                                + rewrite/decomposition, BOTH run
                                unconditionally, merged, deduped, and
                                reranked once. Requires use_routing=True.

    *** REDEFINITION NOTICE *** — use_retrieval_fixes=True now means
    something DIFFERENT from every MLflow run logged under this arm tag
    before this change. Previously (no gate): exact-match, then rewrite
    ONLY, unconditionally — dense search was never consulted except via the
    gate's now-removed dense_confident branch. NOW: exact-match, then dense
    (widened to RERANK_POOL_K) AND rewrite BOTH run, merged, reranked. These
    are genuinely different retrieval mechanisms sharing one flag name.
    RESOLVED: future runs are tagged arm="routed_with_fixes_reranked" (not
    "routed_with_fixes") — see __main__ below — so past and future runs
    aren't silently conflated under identical MLflow tags, same lineage-
    integrity concern RECALL_BASELINE_RUN_ID exists to protect elsewhere in
    this project. Any run still tagged "routed_with_fixes" predates this
    change and used the old (no-dense-fallback) mechanism.

    Three valid arms (was four — routed_with_fixes_gated no longer exists):
        blind:                       use_routing=False, use_retrieval_fixes=False
        routed:                      use_routing=True,  use_retrieval_fixes=False
        routed_with_fixes_reranked:  use_routing=True,  use_retrieval_fixes=True
                                      (see redefinition notice above)

    A scored row wrongly routed to skip retrieves NOTHING (recall 0) — the
    misroute is penalised, not hidden, regardless of use_retrieval_fixes.

    Gating (route-based, three-way — restraint vs misroute vs normal):
        - gated-out = rows with NO expected IDs. Never scored, either arm.
        - scored    = rows WITH expected IDs. A skip-route here is a misroute
                      -> empty retrieval -> recall 0.
        Reconciliation invariant: len(scored) + len(gated_out) == total.

    Returns {"precision_overall": float, "recall_overall": float} on a completed
    run, or None if evaluation did not run.
    """
    if K != PINNED_BASELINE_K:
        raise ValueError(
            f"K={K} does not match the pinned baseline K={PINNED_BASELINE_K}. "
            f"The per-row 'eligible' flags and recall@K would mean something "
            f"different from the baseline the faithfulness gate keys against. "
            f"Re-pin the baseline deliberately or run at K={PINNED_BASELINE_K}."
        )

    if use_routing and router_client is None:
        raise ValueError("use_routing=True requires a router_client.")

    if use_retrieval_fixes and not use_routing:
        raise ValueError(
            "use_retrieval_fixes=True requires use_routing=True — the fixes "
            "(exact-match, dense+rewrite merge, rerank) operate on route "
            "decisions and have no defined behavior against blind "
            "whole-store retrieval."
        )

    # 1. Load the ground-truth dataset
    try:
        with open(dataset_path, 'r', encoding='utf-8') as f:
            all_queries = json.load(f)
    except Exception as e:
        logger.error(f"Failed to load dataset: {e}")
        return None

    total = len(all_queries)

    def _has_expected(q):
        return len(q.get('expected_technique_ids', [])) > 0 or len(q.get('expected_cve_ids', [])) > 0

    scored_queries = [q for q in all_queries if _has_expected(q)]
    gated_out      = [q for q in all_queries if not _has_expected(q)]

    if len(scored_queries) + len(gated_out) != total:
        raise RuntimeError(
            f"Partition failed: {len(scored_queries)} scored + {len(gated_out)} "
            f"gated-out != {total} total."
        )

    if not scored_queries:
        logger.warning("No scored queries (all rows gated out). Nothing to evaluate.")
        return None

    db = ThreatIntelDB()
    corpus_stamp = _corpus_stamp(db)

    metrics_tracker = defaultdict(lambda: {"precision_sum": 0.0, "recall_sum": 0.0, "count": 0})
    per_row_records = []

    logger.info(
        f"Evaluating {len(scored_queries)} scored queries at K={K} "
        f"(use_routing={use_routing}, use_retrieval_fixes={use_retrieval_fixes}, "
        f"{len(gated_out)} gated out)..."
    )
    logger.info(
        f"Corpus stamp: total={corpus_stamp['chunk_count_total']}, "
        f"mitre={corpus_stamp['chunk_count_mitre']}, "
        f"kev={corpus_stamp['chunk_count_kev']} (reconciled)."
    )

    global_precision_sum = 0.0
    global_recall_sum = 0.0
    global_count = 0
    misroute_ids = []

    mlflow.log_metric("eval_dataset_size", len(scored_queries))
    mlflow.log_metric("n_gated_out", len(gated_out))
    mlflow.log_metric("eval_total_rows", total)
    mlflow.log_metric("corpus_chunk_count_total", corpus_stamp["chunk_count_total"])
    mlflow.log_metric("corpus_chunk_count_mitre", corpus_stamp["chunk_count_mitre"])
    mlflow.log_metric("corpus_chunk_count_kev", corpus_stamp["chunk_count_kev"])

    made_live_call = False

    def _throttle():
        nonlocal made_live_call
        if made_live_call:
            time.sleep(THROTTLE_SECONDS)
        made_live_call = True

    # 4. Core Evaluation Loop (scored rows only)
    for query_row in scored_queries:
        expected_ids = set(query_row['expected_technique_ids']) | set(query_row['expected_cve_ids'])

        route_value = None
        retrieval_path = None

        if use_routing:
            _throttle()
            decision = route_query(query_row['query'], router_client)
            route_value = decision.route.value

            if decision.route == Route.SKIP:
                retrieved_ids = []
                retrieval_path = "skip_misroute"
                misroute_ids.append(query_row.get('id'))
                logger.warning(
                    f"MISROUTE {query_row.get('id')}: scored query routed to skip."
                )
            else:
                corpus = _route_to_corpus(decision.route)

                if use_retrieval_fixes:
                    results = retrieve_for_route(
                        db, query_row['query'], router_client, K, corpus,
                        throttle_fn=_throttle,
                    )
                else:
                    results = db.semantic_search(query_row['query'], n_results=K, corpus=corpus)

                if results.get("error"):
                    logger.error(f"Search failed: {query_row['query'][:40]}... {results['error']}")
                    continue
                retrieved_ids = [m['technique_id'] for m in results['metadatas'][0]]
                if use_retrieval_fixes:
                    # retrieve_for_route() sets results["path"] to
                    # "exact_match" or "merged_reranked" — the old
                    # "dense_confident" value no longer exists.
                    retrieval_path = results.get("path", "unknown")
        else:
            # Blind baseline — whole store, no filter, no API call. Unchanged.
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
            "route": route_value,
            "retrieval_path": retrieval_path,
            "retrieved_ids": retrieved_ids,
            "expected_ids": sorted(expected_ids),
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
    print(f"\n--- Retrieval Evaluation Summary (K={K}, use_routing={use_routing}, "
          f"use_retrieval_fixes={use_retrieval_fixes}) ---")
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

    path_counts = {}
    if use_retrieval_fixes:
        path_counts = defaultdict(int)
        for r in per_row_records:
            path_counts[r["retrieval_path"] or "unknown"] += 1
        path_counts = dict(path_counts)
        for path_name, count in sorted(path_counts.items()):
            mlflow.log_metric(f"n_path_{path_name}", count)
        n_merged = path_counts.get("merged_reranked", 0)
        logger.info(
            f"Retrieval path distribution: {path_counts} "
            f"({n_merged}/{len(per_row_records)} queries went through merge+rerank)"
        )

    print(f"\n{'Per-Row Detail':<8} | {'Recall':<7} | {'Retrieved':<40} | Expected")
    print("-" * 100)
    for r in per_row_records:
        flag = "  " if r[f"recall_at_{K}"] > 0 else "!!"
        print(
            f"{flag} {r['id']:<6} | {r[f'recall_at_{K}']:<7.2f} | "
            f"{str(r['retrieved_ids']):<40} | {r['expected_ids']}"
        )
    print("-" * 100)

    # ─── Reconciliation (fail-loud) & Artifact ───
    per_row_recall_sum = sum(r[f"recall_at_{K}"] for r in per_row_records)
    if not math.isclose(per_row_recall_sum, global_recall_sum, rel_tol=1e-9, abs_tol=1e-12):
        raise AssertionError(
            f"Reconciliation FAILED: per-row recall sum ({per_row_recall_sum}) "
            f"!= global recall sum ({global_recall_sum})."
        )
    if len(per_row_records) + len(gated_out) != total:
        raise AssertionError(
            f"Row reconciliation FAILED: {len(per_row_records)} scored + "
            f"{len(gated_out)} gated-out != {total} total. A scored row was "
            f"dropped mid-loop (search error?) — refusing to write the artifact."
        )
    logger.info("Reconciliation OK: recall sums match; scored + gated == total.")

    artifact_payload = {
        "use_routing": use_routing,
        "use_retrieval_fixes": use_retrieval_fixes,
        "retrieval_path_counts": path_counts,
        "corpus_stamp": {
            **corpus_stamp,
            "note": (
                "Per-corpus counts (mitre/kev), reconciled against the "
                "combined total at read time. Content-only edits at "
                "constant per-corpus count are NOT detected; corpus "
                "version is guaranteed by run lineage."
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
    ap.add_argument("--use-retrieval-fixes", action="store_true",
                    help="Requires --use-routing. Exact-match + widened "
                         "dense + rewrite, merged and reranked.")
    args = ap.parse_args()

    if args.use_retrieval_fixes and not args.use_routing:
        ap.error("--use-retrieval-fixes requires --use-routing")

    client = None
    if args.use_routing:
        from gemini_client import get_client
        client = get_client()

    mlflow.set_experiment(config.MLFLOW_EXPERIMENT_NAME)

    if args.use_retrieval_fixes:
        # Renamed from "routed_with_fixes" — that tag now means a
        # structurally different retrieval mechanism (see REDEFINITION
        # NOTICE in run_evaluation()'s docstring). New tag keeps past runs
        # (exact-match + unconditional rewrite, no dense fallback) from
        # being silently conflated with future runs (exact-match + widened
        # dense + rewrite, merged and reranked) under one identical name.
        run_name = "retrieval_eval_routed_fixed_reranked"
        arm_tag = "routed_with_fixes_reranked"
    elif args.use_routing:
        run_name = "retrieval_eval_routed"
        arm_tag = "routed"
    else:
        run_name = "retrieval_eval_blind"
        arm_tag = "blind"

    with mlflow.start_run(run_name=run_name):
        mlflow.set_tag("arm", arm_tag)

        mlflow.log_param("use_routing",         args.use_routing)
        mlflow.log_param("use_retrieval_fixes", args.use_retrieval_fixes)
        if args.use_retrieval_fixes:
            mlflow.log_param("rerank_model",    config.RERANK_MODEL)
            mlflow.log_param("rerank_pool_k",   config.RERANK_POOL_K)
        mlflow.log_param("k",                   config.RETRIEVAL_TOP_K)
        mlflow.log_param("embedding_model",     config.EMBEDDING_MODEL)
        mlflow.log_param("chunk_size",          config.CHUNK_SIZE)
        mlflow.log_param("chunk_overlap",       config.CHUNK_OVERLAP)
        mlflow.log_param("max_chunk_chars",     config.MAX_CHUNK_CHARS)
        if args.use_routing:
            from routing import _ROUTER_MODEL
            mlflow.log_param("router_model", _ROUTER_MODEL)

        run_evaluation(
            str(config.EVAL_SET_PATH),
            use_routing=args.use_routing,
            use_retrieval_fixes=args.use_retrieval_fixes,
            router_client=client,
        )