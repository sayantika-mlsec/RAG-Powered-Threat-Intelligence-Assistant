import os
from pathlib import Path
from dotenv import load_dotenv


load_dotenv(dotenv_path=Path(__file__).parent / ".env", override=True)

GCP_PROJECT_ID = os.environ["GCP_PROJECT_ID"]
GCP_LOCATION = os.environ.get("GCP_LOCATION", "us-central1")

# ─── Data Paths ───────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent
INTEL_DIR = BASE_DIR / "threat_reports"
DB_PATH = BASE_DIR / "brain"
EVAL_SET_PATH = BASE_DIR / "eval_set.json"

# ─── Vector Database & Chunking Limits ────────────────────────────────────────
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
MAX_CHUNK_CHARS = 4000

# ─── Retrieval & Generation Hyperparameters ───────────────────────────────────
RETRIEVAL_TOP_K = 3
MAX_QUERY_LENGTH = 2000

# ─── MLflow Settings ──────────────────────────────────────────────────────────
MLFLOW_EXPERIMENT_NAME = "Threat_Intel_RAG_Evaluation"