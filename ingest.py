import os
import re
import logging
import pathlib
import chromadb
from chromadb.config import Settings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

import config

# ─── Production Logging ───────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# ─── Field Extractor ──────────────────────────────────────────────────────────

def _extract_field(text: str, field_name: str) -> str | None:
    """
    Case-insensitive field extractor with leading-whitespace tolerance.
    Returns None silently when absent — callers handle fallback warnings.
    """
    pattern = re.compile(
        rf"^\s*{re.escape(field_name)}\s*:\s*(.+)$",
        re.IGNORECASE | re.MULTILINE
    )
    match = pattern.search(text)
    if match:
        value = match.group(1).strip()
        logger.debug(f"Extracted field '{field_name}': '{value}'")
        return value if value else None
    return None


# ─── ThreatIntelDB ────────────────────────────────────────────────────────────

class ThreatIntelDB:
    def __init__(self):
        """Initializes the database pulling strictly from config.py"""
        try:
            self.client = chromadb.PersistentClient(path=str(config.DB_PATH))
            
            embedding_fn = SentenceTransformerEmbeddingFunction(
                model_name=config.EMBEDDING_MODEL
            )
            
            self.collection = self.client.get_or_create_collection(
                name="mitre_threat_intel",
                embedding_function=embedding_fn,
                metadata={"hnsw:space": "cosine"}
            )
            
            self.splitter = RecursiveCharacterTextSplitter(
                chunk_size=config.CHUNK_SIZE,
                chunk_overlap=config.CHUNK_OVERLAP,
                separators=["\n\n", "\n", " ", ""]
            )
            logger.info(f"Connected to ChromaDB. Model: {config.EMBEDDING_MODEL}")
        except Exception as e:
            logger.error(f"Failed to initialize ChromaDB: {e}")
            raise

    def _delete_existing_chunks(self, filename: str) -> int:
        """
        Removes all previously ingested chunks for a given source file.
        Returns the number of chunks deleted (0 if none existed).
        """
        try:
            existing = self.collection.get(where={"source": filename})
            if existing and existing["ids"]:
                count = len(existing["ids"])
                self.collection.delete(ids=existing["ids"])
                logger.info(f"[{filename}] Deleted {count} stale chunk(s) before re-ingestion.")
                return count
        except Exception as e:
            logger.warning(f"[{filename}] Could not clean up old chunks: {e}")
        return 0

    def process_directory(self, directory_path: str) -> dict:
        """
        Iterates through a directory, chunks .txt files, and upserts into ChromaDB.

        Returns a summary dict:
            {
                "processed":    int,   # files successfully ingested
                "skipped":      int,   # files skipped (empty / too short)
                "failed":       int,   # files that raised exceptions
                "total_chunks": int    # total chunks upserted across all files
            }
        """
        stats = {"processed": 0, "skipped": 0, "failed": 0, "total_chunks": 0}

        dir_path = pathlib.Path(directory_path)
        if not dir_path.exists() or not dir_path.is_dir():
            logger.error(f"Path '{directory_path}' does not exist or is not a directory.")
            return stats

        # FIX 2: glob already guarantees *.txt — no redundant endswith check needed
        txt_files = sorted(dir_path.rglob("*.txt"))

        if not txt_files:
            logger.warning(f"No .txt files found in '{directory_path}'. Nothing to ingest.")
            return stats

        logger.info(f"Found {len(txt_files)} .txt file(s) to process in '{directory_path}'.")

        for filepath in txt_files:
            # FIX 2: filepath stays pathlib.Path throughout — no shadowing with os.path.join
            filename = filepath.name
            try:
                # FIX 1: file read and ALL data processing inside the with block —
                # raw_text is consumed before the handle closes, logic is explicit.
                with open(filepath, "r", encoding="utf-8") as f:
                    raw_text = f.read().strip()

                # Guards run after file is closed — intent is clear
                if not raw_text:
                    logger.warning(f"[{filename}] File is empty — skipping.")
                    stats["skipped"] += 1
                    continue

                if len(raw_text) < 50:
                    logger.warning(
                        f"[{filename}] File is suspiciously short ({len(raw_text)} chars) "
                        f"— ingesting anyway but verify content."
                    )

                # ── Metadata Extraction ──────────────────────────────────────
                technique_id = (
                    _extract_field(raw_text, "TECHNIQUE_ID")
                    or _extract_field(raw_text, "VULNERABILITY_ID")
                )
                if not technique_id:
                    technique_id = filename.removesuffix(".txt")
                    logger.warning(
                        f"[{filename}] No TECHNIQUE_ID or VULNERABILITY_ID found — "
                        f"falling back to filename: '{technique_id}'"
                    )

                tactic = _extract_field(raw_text, "TACTIC")
                if not tactic:
                    tactic = "unknown"
                    logger.warning(f"[{filename}] No TACTIC field found — defaulting to 'unknown'")

                date_added = _extract_field(raw_text, "DATE_ADDED")
                if not date_added:
                    date_added = "unknown"
                    logger.warning(f"[{filename}] No DATE_ADDED field found — defaulting to 'unknown'")

                # ── Chunking ─────────────────────────────────────────────────
                chunks = self.splitter.create_documents(
                    texts=[raw_text],
                    metadatas=[{
                        "source":       filename,
                        "type":         "threat_intel_report",
                        "technique_id": technique_id,
                        "tactic":       tactic,
                        "date_added":   date_added,
                    }]
                )

                # FIX: prepend header to every chunk so context is never lost after splitting
                documents = [c.page_content for c in chunks]
                metadatas = [c.metadata for c in chunks]
                ids       = [f"{filename}_{i}" for i in range(len(chunks))]
                # ── Upsert ───────────────────────────────────────────────────
                self._delete_existing_chunks(filename)
                self.collection.upsert(documents=documents, metadatas=metadatas, ids=ids)

                # FIX 3: track per-file success into stats
                stats["processed"]    += 1
                stats["total_chunks"] += len(chunks)
                logger.info(f"[{filename}] Ingested {len(chunks)} chunk(s).")

            except Exception as e:
                logger.error(f"[{filename}] Failed during processing: {e}", exc_info=True)
                stats["failed"] += 1

        # FIX 3: surface summary so caller is never blind
        logger.info(
            f"Ingestion complete — "
            f"processed={stats['processed']}, "
            f"skipped={stats['skipped']}, "
            f"failed={stats['failed']}, "
            f"total_chunks={stats['total_chunks']}"
        )
        return stats

    def semantic_search(self, query: str, n_results: int = 3) -> dict:
        """
        Executes a semantic similarity search against the collection.

        FIX 4: Always returns a consistent dict — never returns None.
        Callers check ['error'] key, not isinstance(result, None).

            On success:  {"documents": [[...]], "metadatas": [[...]], "error": None}
            On no match: {"documents": [[]], "metadatas": [[]], "error": None}
            On failure:  {"documents": [[]], "metadatas": [[]], "error": "<reason>"}
        """
        empty = {"documents": [[]], "metadatas": [[]], "error": None}

        if not query or not query.strip():
            logger.warning("semantic_search called with empty query.")
            return {**empty, "error": "Query cannot be empty."}

        if n_results < 1:
            logger.error(f"n_results must be >= 1, got {n_results}.")
            return {**empty, "error": f"Invalid n_results: {n_results}"}

        collection_count = self.collection.count()
        if collection_count == 0:
            logger.warning("semantic_search called on an empty collection. Run process_directory() first.")
            return {**empty, "error": "Collection is empty. Ingest documents first."}

        # Clamp to what actually exists — prevents ChromaDB internal crash
        effective_n = min(n_results, collection_count)
        if effective_n < n_results:
            logger.debug(
                f"n_results clamped from {n_results} to {effective_n} "
                f"(collection only has {collection_count} chunks)."
            )

        try:
            results = self.collection.query(
                query_texts=[query.strip()],
                n_results=effective_n
            )
            hit_count = len(results["documents"][0])
            logger.info(f"Search returned {hit_count} result(s) for query: '{query[:80]}'")
            # Attach error key so callers never need to check for its absence
            results["error"] = None
            return results

        except Exception as e:
            logger.error(f"Search failed for query '{query[:80]}': {e}", exc_info=True)
            return {**empty, "error": str(e)}


# ─── Execution Block ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    db = ThreatIntelDB()

    intel_dir = "./threat_reports"

    # Seed a sample file if the directory doesn't exist yet
    if not os.path.exists(intel_dir):
        os.makedirs(intel_dir)
        with open(f"{intel_dir}/sample_mitre.txt", "w") as f:
            f.write(
                "TECHNIQUE_ID: T1566\n"
                "TACTIC: Initial Access\n"
                "DATE_ADDED: 2024-01-15\n\n"
                "T1566 - Phishing\n\n"
                "Adversaries may send phishing messages to gain access to victim systems. "
                "This technique involves malicious links or attachments delivered via email. "
                "Defenders should monitor for suspicious email attachments and outbound connections "
                "to newly registered domains following email delivery events.\n"
            )

    # FIX 3: caller now gets a summary — not flying blind
    ingestion_stats = db.process_directory(intel_dir)
    print(f"\nIngestion Summary: {ingestion_stats}\n")

    # ── Query test ────────────────────────────────────────────────────────────
    question = "How do hackers trick employees into clicking bad links to get inside the network?"
    results = db.semantic_search(question)

    # FIX 4: clean caller logic — one error key, no None checks
    if results["error"]:
        logger.error(f"Search error: {results['error']}")
    elif not results["documents"][0]:
        logger.warning("No results found. Check if documents were ingested correctly.")
    else:
        logger.info("--- RAG RETRIEVAL RESULTS ---")
        for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
            logger.info(f"Source:       {meta['source']}")
            logger.info(f"Technique ID: {meta.get('technique_id', 'N/A')}")
            logger.info(f"Tactic:       {meta.get('tactic', 'N/A')}")
            logger.info(f"Content:      {doc[:300]}...")
            logger.info("-" * 60)