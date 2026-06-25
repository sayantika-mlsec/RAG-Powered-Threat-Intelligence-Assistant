import json
import logging
import mlflow
from collections import defaultdict
from ingest import ThreatIntelDB


import config

# ─── Logging Setup ────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# ─── Evaluation Pipeline ──────────────────────────────────────────────────────

def run_evaluation(dataset_path: str, K: int = config.RETRIEVAL_TOP_K):
    """
    Evaluates the RAG pipeline using Precision@K and Recall@K.
    Requires ingest.py to be present in the same directory.
    """
    # 1. Load the ground-truth dataset
    try:
        with open(dataset_path, 'r', encoding='utf-8') as f:
            all_queries = json.load(f)
    except Exception as e:
        logger.error(f"Failed to load dataset: {e}")
        return

    
    # 2. Filter for relevant categories (Categories 1, 2, 3)
    # Group A queries must have at least one expected ID populated in the ground truth

    group_a_queries = [
        q for q in all_queries 
        if len(q.get('expected_technique_ids', [])) > 0 or len(q.get('expected_cve_ids', [])) > 0
    ]
    
    if not group_a_queries:
        logger.warning("No valid queries found for evaluation.")
        return

    # 3. Initialize ChromaDB Connection
    db = ThreatIntelDB()
    
    # Dictionary to aggregate metrics: {(category, difficulty): {"precision_sum": 0, "recall_sum": 0, "count": 0}}
    metrics_tracker = defaultdict(lambda: {"precision_sum": 0.0, "recall_sum": 0.0, "count": 0})

    logger.info(f"Starting evaluation of {len(group_a_queries)} queries at K={K}...")

    # ── GLOBAL HEADLINE TRACKERS ──
    global_precision_sum = 0.0
    global_recall_sum = 0.0
    global_count = 0

    # Log dataset size to the active MLflow run
    mlflow.log_metric("eval_dataset_size", len(group_a_queries))

    # 4. Core Evaluation Loop
    for query_row in group_a_queries:
        
        # Merge expected MITRE and CVE IDs into a single set
        expected_ids = set(query_row['expected_technique_ids']) | set(query_row['expected_cve_ids'])
        
        # Guard clause: If expected lists are empty, skip to prevent division-by-zero
        if not expected_ids:
            logger.debug(f"Skipping query (empty expected_ids): '{query_row['query'][:40]}...'")
            continue

        # Execute search via semantic_search wrapper method
        results = db.semantic_search(query_row['query'], n_results=K)

        # Safety check: If the wrapper returned an error key, handle it safely
        if results.get("error"):
            logger.error(f"Search failed for query: {query_row['query'][:40]}... Error: {results['error']}")
            continue
        
        # Extract the technique_id from the returned metadata
        retrieved_ids = [m['technique_id'] for m in results['metadatas'][0]]

        print(f"Query: {query_row['query']}")
        print(f"Expected: {expected_ids}")
        print(f"Retrieved: {retrieved_ids}")
        
        # ── Metric Calculation ──
        # Intersection of retrieved IDs and expected IDs
        relevant_retrieved_count = len(set(retrieved_ids) & expected_ids)
        
        precision = relevant_retrieved_count / len(retrieved_ids) if retrieved_ids else 0.0
        
        recall = relevant_retrieved_count / len(expected_ids)
        
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

    # ── Final Headline Calculation ──
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

        # These will attach to the run started in main.py
        mlflow.log_metric(f"precision_{cat}_{diff}", avg_precision)
        mlflow.log_metric(f"recall_{cat}_{diff}", avg_recall)
        
    print("-" * 55)

    # ── Log Headline Metrics to MLflow ──
    mlflow.log_metric("precision_overall", precision_overall)
    mlflow.log_metric("recall_overall", recall_overall)

if __name__ == "__main__":
    # Example usage:
    run_evaluation(str(config.EVAL_SET_PATH))
    pass