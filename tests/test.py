import pytest
from ingest import ThreatIntelDB

# ─── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture
def sample_intel_dir(tmp_path):
    """
    Creates a temporary directory and writes a strictly formatted 
    MITRE sample file into it. Pytest deletes this when the test ends.
    """
    intel_dir = tmp_path / "threat_reports"
    intel_dir.mkdir()
    
    sample_file = intel_dir / "T1566.txt"
    sample_file.write_text(
        "TECHNIQUE_ID: T1566\n"
        "TACTIC: Initial Access\n"
        "DATE_ADDED: 2024-01-15\n\n"
        "T1566 - Phishing\n\n"
        "Adversaries may send phishing messages to gain access to victim systems. "
        "This technique involves malicious links or attachments delivered via email. "
        "Defenders should monitor for suspicious email attachments.",
        encoding="utf-8"
    )
    
    return str(intel_dir)

@pytest.fixture
def isolated_db(tmp_path):
    """
    Initializes ThreatIntelDB pointing to an ephemeral ChromaDB instance 
    inside the tmp_path. Ensures tests do not pollute the production DB.
    """
    db_dir = tmp_path / "test_brain"
    # Using the standard defaults, but isolating the storage path
    return ThreatIntelDB(db_path=str(db_dir))


# ─── Tests ────────────────────────────────────────────────────────────────────

def test_ingestion_smoke(isolated_db, sample_intel_dir):
    """
    Test 1: Ingestion smoke test.
    Verifies that process_directory reads the file, parses it, and 
    successfully upserts chunks without error.
    """
    stats = isolated_db.process_directory(sample_intel_dir)
    
    assert stats["processed"] == 1
    assert stats["failed"] == 0
    assert stats["skipped"] == 0
    assert stats["total_chunks"] > 0

def test_retrieval_smoke(isolated_db, sample_intel_dir):
    """
    Test 2: Retrieval smoke test.
    Verifies that after ingestion, a semantic query successfully returns 
    the relevant chunk with the correct metadata and no errors.
    """
    # Setup: Run the ingestion phase first
    isolated_db.process_directory(sample_intel_dir)
    
    # Execution: Query the database
    query = "How do adversaries use email attachments?"
    results = isolated_db.semantic_search(query, n_results=1)
    
    # Assertions: Verify the contract and content
    assert results["error"] is None, "semantic_search returned an unexpected error"
    
    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]
    print(documents[0])
    
    assert len(documents) > 0, "Expected at least one document chunk, got none"
    assert "phishing" in documents[0].lower(), "Document chunk lacks expected content"
    assert metadatas[0]["technique_id"] == "T1566", "Metadata technique_id mismatch"