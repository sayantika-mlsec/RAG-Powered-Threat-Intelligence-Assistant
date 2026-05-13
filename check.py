from ingest import ThreatIntelDB

db = ThreatIntelDB()
question = "How do adversaries use phishing for initial access?"
results = db.semantic_search(question, n_results=3)

print("Total chunks in DB:", db.collection.count())
print("Chunks retrieved:", len(results["documents"][0]))

for i, (doc, meta) in enumerate(zip(results["documents"][0], results["metadatas"][0])):
    print(f"\n=== CHUNK {i+1} ===")
    print("TECHNIQUE_ID:", meta.get("technique_id"))
    print("TACTIC:", meta.get("tactic"))
    print("FIRST 100 CHARS:", doc[:100])
    print("LAST 100 CHARS:", doc[-100:])