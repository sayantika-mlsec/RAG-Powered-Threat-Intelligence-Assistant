import json
import logging
from dataclasses import dataclass
from google.genai import types as genai_types

logger = logging.getLogger(__name__)

_REWRITE_MODEL = "gemini-2.5-flash"

_REWRITE_SCHEMA = genai_types.Schema(
    type=genai_types.Type.OBJECT,
    properties={
        "sub_queries": genai_types.Schema(
            type=genai_types.Type.ARRAY,
            items=genai_types.Schema(
                type=genai_types.Type.OBJECT,
                properties={
                    "text": genai_types.Schema(type=genai_types.Type.STRING),
                    "corpus": genai_types.Schema(
                        type=genai_types.Type.STRING,
                        enum=["mitre", "kev"],
                    ),
                },
                required=["text", "corpus"],
            ),
        )
    },
    required=["sub_queries"],
)

_SYSTEM_INSTRUCTION = (
    "You rewrite plain-English security questions into technically-phrased "
    "retrieval queries for a vector database containing two corpora: "
    "'mitre' (MITRE ATT&CK techniques) and 'kev' (CISA Known Exploited "
    "Vulnerabilities).\n\n"
    "Rules:\n"
    "1. Single concept -> return exactly ONE rewritten query, with the "
    "corpus it belongs to. Do NOT split a query into multiple sub-queries "
    "just because it contains a word that resembles a different MITRE "
    "tactic or technique name (e.g. a query about hiding C2 traffic is "
    "ONE technique about obfuscation — 'command and control' here is "
    "context describing what's being hidden, not a second technique to "
    "search for separately).\n"
    "2. Query chains MULTIPLE genuinely distinct, sequential ACTIONS an "
    "adversary performs (e.g. 'bypass MFA, THEN log in, THEN modify "
    "settings') or spans both a vulnerability and a technique -> split "
    "into one sub-query PER distinct action, each tagged with its own "
    "correct corpus. A query describing ONE technique using several "
    "related terms is NOT this case.\n"
    "3. For 'mitre' sub-queries: phrase as close as possible to official "
    "MITRE ATT&CK technique naming conventions (e.g. 'Exploit Public-Facing "
    "Application', 'Valid Accounts', 'Data Obfuscation') rather than generic "
    "paraphrases — MITRE's own vocabulary retrieves far better than loose "
    "synonyms ('public-facing' != 'internet-facing' to this retriever).\n"
    "4. Prefer the PARENT technique name over a specific sub-technique "
    "unless the query names a specific implementation detail. A query "
    "describing a general category of behavior (e.g. 'manipulating search "
    "orders, environment variables, OR path locations' — several examples "
    "of the same general technique) wants the parent technique, not one "
    "narrow sub-technique that matches only one of the examples.\n"
    "5. Never invent technique names or CVE numbers not implied by the "
    "original query.\n"
    "6. Return 1 to 4 sub-queries. Default to 1 unless the query genuinely "
    "chains multiple distinct actions or corpora per Rule 2."
)


@dataclass
class SubQuery:
    text: str
    corpus: str | None  # None only on the fallback-failure path


def rewrite_query(query: str, client, throttle_fn=None) -> list[SubQuery]:
    """
    Rewrites a query into 1+ technically-phrased, corpus-tagged retrieval
    queries. Pure function apart from the throttle hook: no DB access, no
    retrieval — one Gemini call total, regardless of how many sub-queries
    come back.

    throttle_fn: optional zero-arg callable invoked immediately before the
    live API call. Lets a caller (e.g. the eval harness) enforce a single
    shared rate-limit gap across BOTH routing calls and rewrite calls,
    rather than each call site throttling independently and under-counting
    real request volume against the daily quota.

    Fails open: any error returns [SubQuery(query, None)] — corpus=None
    signals "no prediction, search unfiltered" to the caller.
    """
    if throttle_fn is not None:
        throttle_fn()

    try:
        response = client.models.generate_content(
            model=_REWRITE_MODEL,
            contents=query,
            config=genai_types.GenerateContentConfig(
                system_instruction=_SYSTEM_INSTRUCTION,
                temperature=0.0,
                response_mime_type="application/json",
                response_schema=_REWRITE_SCHEMA,
                max_output_tokens=512,
                thinking_config=genai_types.ThinkingConfig(thinking_budget=0),
            ),
        )
        parsed = json.loads(response.text)
        raw_sub_queries = parsed.get("sub_queries") or [{"text": query, "corpus": None}]
        sub_queries = [SubQuery(text=sq["text"], corpus=sq.get("corpus")) for sq in raw_sub_queries]
        logger.info(f"Rewrote '{query[:50]}...' -> {sub_queries}")
        return sub_queries
    except Exception as e:
        logger.warning(f"Query rewrite failed, falling back to original: {e}")
        return [SubQuery(text=query, corpus=None)]


def retrieve_with_rewrite(
    db,
    query: str,
    client,
    n_results: int,
    corpus: str | None = None,
    throttle_fn=None,
) -> dict:
    """
    Orchestrator — rewrite_query() -> per-sub-query retrieval -> guaranteed-
    slot merge. Same return contract as ThreatIntelDB.semantic_search().

    NOTE: each sub-query fetches exactly n_results candidates (not a widened
    pool). Pool-widening (fetching more than n_results per sub-query so a
    lower-ranked correct answer has a chance to surface during merge) was
    investigated as a possible additional fix for q015/q016 but found
    insufficient on its own (confirmed: T1190 sat at rank 13/15 even with
    canonical MITRE phrasing) and was NOT adopted — those two residual
    misses are tracked as a documented limitation instead of solved here.
    Do not silently widen this without re-opening that scope decision.

    Corpus resolution per sub-query:
      - Caller passed an explicit corpus ("mitre"/"kev" — single-corpus
        route already decided upstream by the router) -> that filter wins
        for every sub-query, overriding any tag.
      - Caller passed corpus=None (BOTH route) -> each sub-query uses its
        OWN predicted corpus tag, letting a cross-collection query route
        its KEV-flavored half and MITRE-flavored half independently.
      - Sub-query's own tag is also None (rewrite failure fallback) ->
        that sub-query searches unfiltered.

    Merge strategy — guaranteed slot per sub-query, then fill globally:
      Raw distances from different sub-queries are NOT on a comparable
      scale (confirmed empirically: a wrong-but-tight KEV cluster can sit
      at a lower absolute distance than a correct MITRE match). Pure
      global top-K by distance can let one sub-query's results crowd out
      another's correct answer entirely. Guaranteeing each sub-query's own
      best result a seat first prevents that starvation.
    """
    sub_queries = rewrite_query(query, client, throttle_fn=throttle_fn)

    per_subquery_results: list[list[tuple[float, dict, str]]] = []
    for sq in sub_queries:
        effective_corpus = corpus if corpus is not None else sq.corpus
        result = db.semantic_search(sq.text, n_results=n_results, corpus=effective_corpus)
        if result.get("error"):
            logger.warning(f"Sub-query search failed: '{sq.text[:50]}...' {result['error']}")
            per_subquery_results.append([])
            continue
        docs = result["documents"][0]
        metas = result["metadatas"][0]
        dists = result.get("distances", [[]])[0]
        ranked_sq = sorted(zip(dists, metas, docs), key=lambda c: c[0])
        per_subquery_results.append(ranked_sq)

    if not any(per_subquery_results):
        return {"documents": [[]], "metadatas": [[]], "distances": [[]], "error": None}

    # Guarantee slot 1: best result from EACH sub-query, deduped by technique_id
    seen: set[str] = set()
    final: list[tuple[float, dict, str]] = []
    for ranked_sq in per_subquery_results:
        for dist, meta, doc in ranked_sq:
            tid = meta.get("technique_id")
            if tid not in seen:
                seen.add(tid)
                final.append((dist, meta, doc))
                break  # one guaranteed slot per sub-query, take its best

    # Fill remaining slots globally, by distance, from whatever's left
    remaining_pool = [
        (dist, meta, doc)
        for ranked_sq in per_subquery_results
        for dist, meta, doc in ranked_sq
        if meta.get("technique_id") not in seen
    ]
    remaining_pool.sort(key=lambda c: c[0])

    for dist, meta, doc in remaining_pool:
        if len(final) >= n_results:
            break
        tid = meta.get("technique_id")
        if tid not in seen:
            seen.add(tid)
            final.append((dist, meta, doc))

    final = final[:n_results]
    return {
        "documents": [[d for _, _, d in final]],
        "metadatas": [[m for _, m, _ in final]],
        "distances": [[dist for dist, _, _ in final]],
        "error": None,
    }