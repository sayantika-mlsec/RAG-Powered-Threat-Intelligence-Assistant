from exact_id import extract_exact_ids
from query_rewrite import retrieve_with_rewrite


def retrieve_for_route(db, query: str, client, k: int, corpus: str | None, throttle_fn=None) -> dict:
    """
    corpus="mitre" / "kev" -> single-corpus route.
    corpus=None -> BOTH route, sub-queries corpus-routed individually.

    Step 1 — exact match: collects ALL literal CVE/technique IDs in the
    raw query (any count, any corpus mix), fetches each directly. Local
    ChromaDB metadata lookup — never throttled, no API call.
    Step 2 — rewrite fallback: only runs if no literal IDs were found, or
    none of the found IDs matched actual stored data. Makes a live Gemini
    call — throttle_fn (if provided) is invoked immediately before it, so
    a caller sharing one throttle across routing + rewrite calls (e.g. the
    eval harness) enforces a single consistent rate-limit gap rather than
    each call site throttling independently.
    """
    exact_ids = extract_exact_ids(query)

    if exact_ids:
        documents, metadatas = [], []
        for exact_id, exact_corpus in exact_ids:
            result = db.collection.get(
                where={
                    "$and": [
                        {"corpus": {"$eq": exact_corpus}},
                        {"technique_id": {"$eq": exact_id}},
                    ]
                },
                include=["documents", "metadatas"],
            )
            documents.extend(result["documents"])
            metadatas.extend(result["metadatas"])

        if documents:
            n = min(k, len(documents))
            return {
                "documents": [documents[:n]],
                "metadatas": [metadatas[:n]],
                "distances": [[0.0] * n],
                "error": None,
            }
        # IDs were present in the text but matched nothing in storage —
        # fall through to rewrite rather than returning empty.

    return retrieve_with_rewrite(
        db, query, client, n_results=k, corpus=corpus, throttle_fn=throttle_fn,
    )