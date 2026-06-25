import os
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

if __name__ == "__main__":
    # Ensure the experiment exists (or creates it)
    mlflow.set_experiment(config.MLFLOW_EXPERIMENT_NAME)
    
    # Start the run and apply your evaluation milestone tag
    with mlflow.start_run(tags={"milestone": "baseline-with-evaluation"}):
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
        logger.info("Starting Retrieval Evaluation...")
        dataset_path = config.EVAL_SET_PATH 
        
        # The function will automatically log metrics to this active run
        run_evaluation(dataset_path, K=config.RETRIEVAL_TOP_K)
        
    logger.info("Run complete. Open the MLflow UI to view parameters and metrics.")