import os
import math
import logging
import mlflow
from pathlib import Path
import config
from dotenv import load_dotenv
from eval_retrieval import run_evaluation

# ── Environment ───────────────────────────────────────────────────────────────
load_dotenv(dotenv_path=Path(__file__).parent / ".env", override=True)

# Retrieve the URI from the environment
tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
if not tracking_uri:
    raise RuntimeError("MLFLOW_TRACKING_URI not set in .env")

mlflow.set_tracking_uri(tracking_uri)

# ── Logging Setup ─────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# ── Pinned Baseline (of record) ───────────────────────────────────────────────
# This run REPRODUCES and VERIFIES the pinned baseline; it does not re-pin it.
# Values rounded to 4 dp, so verification uses a tolerance (the true value has a
# tail past 0.2444 — exact equality would false-fail).
PINNED_BASELINE = {
    "precision_overall": 0.2444,
    "recall_overall": 0.5667,
}
BASELINE_ABS_TOL = 5e-5

if __name__ == "__main__":
    # Ensure the experiment exists (or create it)
    mlflow.set_experiment(config.MLFLOW_EXPERIMENT_NAME)

    # Tags distinguish this reproduction run from the pinned baseline of record,
    # so two runs with identical numbers are never ambiguous in the UI.
    with mlflow.start_run(tags={
        "milestone": "baseline-with-evaluation",
        "run_purpose": "reproduce-verify-pinned-baseline",
        "reproduces_pinned": "true",
    }):
        logger.info(f"Connected to MLflow tracking server at: {tracking_uri}")
        logger.info("Starting MLflow RAG Pipeline run...")

        # Log all experiment-relevant parameters from the config
        mlflow.log_params({
            "chunk_size": config.CHUNK_SIZE,
            "chunk_overlap": config.CHUNK_OVERLAP,
            "embedding_model": config.EMBEDDING_MODEL,
            "retrieval_top_k": config.RETRIEVAL_TOP_K,
            "max_chunk_chars": config.MAX_CHUNK_CHARS,
            "max_query_length": config.MAX_QUERY_LENGTH,
        })
        logger.info("Successfully logged parameters to MLflow.")

        # ── Execute Evaluation ──
        # run_evaluation also logs granular per-row artifacts/metrics to this run,
        # and raises on its own fail-loud invariants (K-pin, corpus stamp,
        # reconciliation) — those propagate here and mark the run FAILED.
        logger.info("Starting Retrieval Evaluation...")
        dataset_path = config.EVAL_SET_PATH

        result = run_evaluation(str(dataset_path), K=config.RETRIEVAL_TOP_K)

        # ── Verify reproduction against the pinned baseline ──
        # Asserts on the values just computed (returned in-hand), not re-read from
        # MLflow — single source, no cross-sourcing. Drift fails the run loudly.
        if result is None:
            raise RuntimeError(
                "run_evaluation returned None — evaluation did not complete "
                "(empty dataset or load failure). Cannot verify baseline."
            )

        for metric, pinned_val in PINNED_BASELINE.items():
            actual = result[metric]
            if not math.isclose(actual, pinned_val, abs_tol=BASELINE_ABS_TOL):
                raise AssertionError(
                    f"Baseline reproduction FAILED on {metric}: "
                    f"got {actual:.4f}, pinned {pinned_val}. Corpus or pipeline "
                    f"drifted — investigate before trusting this run."
                )
            logger.info(f"Baseline verified: {metric}={actual:.4f} matches pin ({pinned_val}).")

        logger.info("Baseline reproduction verified against pin.")

    logger.info("Run complete. Open the MLflow UI to view parameters and metrics.")