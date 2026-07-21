from pathlib import Path

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

# ─── Reranking (replaces the retired confidence gate) ─────────────────────────
# RERANK_POOL_K: candidate pool width BEFORE reranking (dense search AND the
# rewrite path are each widened to this). Final return size is still governed
# by the `k` argument passed into retrieve_for_route() — pinned to
# RETRIEVAL_TOP_K (3) in eval_retrieval.py, and set independently in app.py.
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-12-v2"
# Was: cross-encoder/ms-marco-MiniLM-L-6-v2. Swapped 2026-07-21 to test
# whether q010/q015's sibling-technique confusion (T1570 beating T1105,
# T1189 beating T1190 — both by wide margins, not near-ties) is a capacity
# ceiling in the L-6 model specifically, or a more fundamental limitation
# of this reranking approach. Same MS MARCO family/training data as L-6 —
# deliberately NOT jumping to a different architecture (e.g.
# BAAI/bge-reranker-base) yet, so this test isolates ONE variable
# (capacity) rather than changing training data and architecture at once.
# If L-12 doesn't resolve it, bge-reranker-base is the next real test —
# not skipped, sequenced second.
RERANK_POOL_K = 15

# ─── MLflow Settings ──────────────────────────────────────────────────────────
MLFLOW_EXPERIMENT_NAME = "Threat_Intel_RAG_Evaluation"

# REMOVED: RETRIEVAL_CONFIDENCE_THRESHOLD = 0.3146
# The confidence gate is retired (see retrieval_pipeline.py module
# docstring). This value has no reader anywhere in the codebase as of this
# change — confirm with a repo-wide grep for RETRIEVAL_CONFIDENCE_THRESHOLD
# before deleting confidence_gate.py / calibrate_confidence.py, in case
# anything outside the four files reviewed in this session still imports it.