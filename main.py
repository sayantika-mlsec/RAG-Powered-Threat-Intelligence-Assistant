import logging
import mlflow
import config

mlflow.set_tracking_uri("http://127.0.0.1:5000")

# ── Logging Setup ─────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

if __name__ == "__main__":
    # Ensure the experiment exists (or creates it)
    mlflow.set_experiment(config.MLFLOW_EXPERIMENT_NAME)
    
    # Start the run and apply your baseline tag
    with mlflow.start_run(tags={"milestone": "baseline-config-no-eval-yet"}):
        logger.info("Starting MLflow plumbing test run...")
        
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
        
    # The run ends cleanly and automatically when exiting the 'with' block
    logger.info("Run complete. Open the MLFlow UI to verify your baseline is tracked.")