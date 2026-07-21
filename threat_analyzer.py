import os
import html
import logging
import time
from dataclasses import dataclass, field
from dotenv import load_dotenv
from pathlib import Path

from google.genai import types as genai_types
from google.genai import errors as genai_errors

from gemini_client import get_client
from routing import ModelTier

# ── Environment ───────────────────────────────────────────────────────────────
# Loaded BEFORE `import config`, not after. This file is imported by app.py via
# `from threat_analyzer import ThreatAnalyzer` — and that import happens BEFORE
# app.py calls its own load_dotenv(). So the very first time `config` gets
# imported anywhere in the whole process is right here, with nothing having
# loaded .env yet. Since config.py now reads GCP_PROJECT_ID at import time
# (Vertex migration), `import config` below would fail before the Gradio app
# even finishes starting up. Loading .env first, in this file, removes that
# dependency on being imported in a lucky order.
load_dotenv(dotenv_path=Path(__file__).parent / ".env", override=True)

import config

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s"
)
logger = logging.getLogger(__name__)


# Model name as env variable — changing models in production = update .env, not code
MODEL_NAME = os.getenv("GEMINI_MODEL_NAME", "gemini-2.5-flash")

# Second tier's model — pro dispatch is new as of tiered routing (Jul 23-25).
# Its own env var, not derived from MODEL_NAME, since the two tiers are
# independently swappable in production.
PRO_MODEL_NAME = os.getenv("GEMINI_PRO_MODEL_NAME", "gemini-2.5-pro")

# Maps a routing decision's tier directly to the model string used for that
# generation call. FLASH resolves to MODEL_NAME — the same model that was
# hardcoded before tiering existed — so a query with no tier info (the blind
# arm, or any caller not yet passing one) is byte-identical to pre-tiering
# behavior. See generate_answer()'s tier=None default below.
_TIER_MODEL = {
    ModelTier.FLASH: MODEL_NAME,
    ModelTier.PRO: PRO_MODEL_NAME,
}

SYSTEM_INSTRUCTION = (
    "You are a Senior SOC Analyst Assistant. "
    "Answer ONLY using the threat intelligence provided in the <threat_intelligence> tags. "
    "The context may contain multiple partial excerpts from threat reports — "
    "synthesize across all of them to form your answer. "
    "If the answer genuinely cannot be found in any part of the provided context, "
    "respond exactly with: "
    "'I do not have sufficient information in the provided threat intelligence.' "
    "Do not speculate. Do not use prior knowledge. Do not invent IOCs, techniques, or sources."
)

# System instruction now travels WITH the generation config, passed per call —
# the new SDK has no persistent "model" object to bake it into (see __init__).
GENERATION_CONFIG = genai_types.GenerateContentConfig(
    system_instruction=SYSTEM_INSTRUCTION,
    temperature=0.0,
    # top_p and top_k intentionally omitted:
    # at temperature=0.0 greedy decoding is active — sampling parameters
    # have no effect and setting them misleads future maintainers.
    #
    # NOTE: temperature=0.0 produces near-deterministic output but is NOT
    # guaranteed deterministic due to floating-point ops across GPU cores.
    max_output_tokens=4096,    # was 2048 — q015 hit 2044/2048 (thinking 1966 +
                               # output 78) and was cut off mid-sentence.
                               # Doubling gives real headroom for compound
                               # cross-corpus queries without touching
                               # anything else — cost is governed by actual
                               # usage, not this ceiling.
    candidate_count=1,
)


# ── Return Contract ───────────────────────────────────────────────────────────

@dataclass
class AnalysisResult:
    answer:           str
    success:          bool
    source_citations: list[dict] = field(default_factory=list)
    error:            str | None = None
    context_used:     list[str] = field(default_factory=list)
    mode:             str = "retrieved"   # "retrieved" | "no_retrieval"
    # Usage/timing — populated by generate_answer() after _call_llm() returns.
    # None on the no_retrieval (skip) path, since that path never calls
    # ThreatAnalyzer at all (see app.py's _no_retrieval_response) — tier
    # dispatch was already forced to flash there and isn't part of what
    # the cost/latency table is measuring.
    latency_seconds:  float | None = None
    input_tokens:     int | None = None
    output_tokens:    int | None = None
    thinking_tokens:  int | None = None

    @classmethod
    def ok(cls, answer: str, citations: list[dict], context_used: list[str]) -> "AnalysisResult":
        return cls(answer=answer, success=True, source_citations=citations, context_used=context_used)

    @classmethod
    def fail(cls, error: str) -> "AnalysisResult":
        return cls(answer="", success=False, source_citations=[], error=error)

    @classmethod
    def no_retrieval(cls, answer: str) -> "AnalysisResult":
        """Skip-route success: answered directly, no context retrieved.
        success=True (not a failure) but mode flags the absence of grounding."""
        return cls(answer=answer, success=True, source_citations=[], context_used=[], mode="no_retrieval")

# ── Module-level prompt utilities ─────────────────────────────────────────────
# Defined at module level, NOT inside generate_answer():
# - redefining functions on every call wastes memory
# - module-level functions are independently testable
# - signals correct understanding of Python scoping

def _sanitize_for_prompt(text: str) -> str:
    """
    Sanitizes USER INPUT only — not internal context chunks.
    Applied exclusively to the query string in _build_prompt().
    MITRE/CISA chunks are trusted internal data and must not be
    escaped — doing so garbles markdown, URLs, and ATT&CK
    cross-references that the LLM needs to read correctly.
    """
    text = text.replace("\x00", "")
    return html.escape(text, quote=True)


def _truncate_chunks(
    chunks: list[str],
    metadatas: list[dict]
) -> tuple[list[str], list[dict]]:
    """
    FIX 3 (truncation): Enforces per-chunk and total-chunk limits BEFORE
    building the prompt string.

    Why chunk-level instead of string-level truncation:
      - String-level truncation (old approach) cuts mid-chunk silently.
        Metadatas still claim N sources contributed, but source N may have
        been entirely discarded. UI would cite a source that contributed
        zero content — a correctness lie.
      - Chunk-level truncation keeps metadatas and chunks in sync:
        every cited source actually contributed text to the answer.

    Returns (truncated_chunks, matching_metadatas) — always the same length.

    Truncates chunks dynamically by pulling limits directly from the global config.
    """
    result_chunks: list[str]  = []
    result_metas:  list[dict] = []

    # Pull limits directly from config
    max_chunks = config.RETRIEVAL_TOP_K
    max_chars  = config.MAX_CHUNK_CHARS

    # Enforce the chunk count limit
    for chunk, meta in zip(chunks[:max_chunks], metadatas[:max_chunks]):

        # Enforce the character limit per chunk
        if len(chunk) > max_chars:
            chunk = chunk[:max_chars]
            logger.warning(
                f"Chunk from '{meta.get('source', 'unknown')}' truncated to "
                f"{max_chars} chars."
            )

        result_chunks.append(chunk)
        result_metas.append(meta)

    return result_chunks, result_metas

def _build_prompt(query: str, context_chunks: list[str]) -> str:
    """
    Only the user query is sanitized — it's the untrusted input.
    Context chunks come from your own ingested MITRE/CISA data —
    sanitizing them garbles markdown, URLs, and special characters
    that the LLM needs to read correctly.
    """
    safe_query = _sanitize_for_prompt(query.strip())

    # Chunks are NOT sanitized — they are trusted internal data
    context_text = "\n\n---\n\n".join(chunk for chunk in context_chunks)

    return (
        "<threat_intelligence>\n"
        f"{context_text}\n"
        "</threat_intelligence>\n"
        "<user_query>\n"
        f"{safe_query}\n"
        "</user_query>"
    )


def _safe_extract_text(response) -> tuple[str | None, str | None]:
    """
    Safely extracts text from a Gemini response object (new google.genai SDK).
    Returns (text, error_reason) — exactly one will be None.

    Checks prompt-level blocks, then candidate-level finish reasons, before
    touching .text — same failure taxonomy as before, adapted to the new
    response shape (defensive getattr, since prompt_feedback may be absent
    entirely when nothing was blocked, rather than present-but-empty).
    """
    prompt_feedback = getattr(response, "prompt_feedback", None)
    block_reason = getattr(prompt_feedback, "block_reason", None) if prompt_feedback else None
    if block_reason:
        return None, f"PROMPT_BLOCKED:{block_reason.name}"

    if not response.candidates:
        return None, "NO_CANDIDATES"

    finish_reason = response.candidates[0].finish_reason
    finish_reason_name = finish_reason.name if finish_reason else "UNKNOWN"

    if finish_reason_name == "SAFETY":
        return None, "CANDIDATE_SAFETY_FILTERED"

    if finish_reason_name == "MAX_TOKENS":
        logger.warning(
            f"Generation for query hit MAX_TOKENS — response may "
            f"be truncated mid-sentence. Returning partial text rather than "
            f"failing, but this should be checked."
        )
        return response.text, None

    if finish_reason_name != "STOP":
        return None, f"UNEXPECTED_FINISH:{finish_reason_name}"

    return response.text, None


# ── ThreatAnalyzer ────────────────────────────────────────────────────────────

class ThreatAnalyzer:
    """
    LLM generation layer for the RAG-powered Threat Intelligence Assistant.

    Class design (vs module-level initialization):
      - Instantiated explicitly — testable and mockable
      - self.client can be swapped in tests without module-level patching
      - Model name and config changes are isolated to __init__ / module constants
    """

    def __init__(self):
        """
        Initializes the analyzer on the shared Vertex-backed client.

        No API key, no genai.configure() — get_client() carries Vertex auth
        (project + ADC) internally. There is also no persistent "model" object
        in the new SDK the way there was in the legacy one: system_instruction
        and generation params now travel per-call inside GENERATION_CONFIG
        (see module level), not baked into an object here.
        """
        self.client = get_client()

    # ── Input validation ──────────────────────────────────────────────────────

    def _validate_query(self, query: str | None) -> tuple[str | None, str | None]:
        """
        Validates and normalises the raw query string.
        Returns (clean_query, error_message) — exactly one will be None.
        """
        if not query or not isinstance(query, str):
            return None, "Invalid query."

        query = query.strip()
        if len(query) < 3:
            return None, "Query too short."

        # Pull directly from config
        if len(query) > config.MAX_QUERY_LENGTH:
            query = query[:config.MAX_QUERY_LENGTH]

        return query, None

    def _validate_search_results(
        self, search_results: dict | None
    ) -> tuple[tuple[list, list] | None, str | None]:
        """
        FIX 1 + FIX 3: Validates the search_results dict from semantic_search()
        and returns BOTH documents and metadatas so citations reach the caller.

        Returns ((chunks, metadatas), error_message) — exactly one will be None.

        Four failure modes distinguished:
          1. None         — caller passed None (contract violation)
          2. error key    — upstream semantic_search() failed (new contract)
          3. Malformed    — unexpected ChromaDB structure
          4. Empty result — search succeeded, no relevant chunks found
        """
        # Guard 1: None — caller violated the dict contract
        if search_results is None:
            logger.warning("generate_answer received None — caller violated contract.")
            return None, "Threat intelligence search failed. Cannot generate analysis."

        # FIX 1: Guard 2 — consume the "error" key from the new ingest.py contract
        # The old code ignored this key entirely, causing wrong error messages in the UI.
        upstream_error = search_results.get("error")
        if upstream_error:
            logger.warning(f"Upstream search error propagated: '{upstream_error}'")
            return None, f"Threat intelligence search failed: {upstream_error}"

        # Guard 3: Safe structural extraction
        try:
            context_chunks = search_results.get("documents", [[]])[0]
            # FIX 3: extract metadatas in sync with chunks — never discard them
            metadatas      = search_results.get("metadatas", [[]])[0]
        except (IndexError, TypeError, AttributeError) as e:
            logger.error(
                f"Malformed search_results structure: {e}. "
                f"Got type: {type(search_results)}"
            )
            return None, "Threat intelligence search returned malformed data."

        # Guard 4: Empty result set — do NOT call the LLM with no context
        if not context_chunks:
            logger.warning("RAG BYPASS: No relevant chunks found. Aborting LLM call.")
            return None, "No relevant threat intel found for this query."

        return (context_chunks, metadatas), None

    # ── Core generation ───────────────────────────────────────────────────────

    def _call_llm(self, prompt: str, query: str, model_name: str) -> tuple[str | None, str | None, dict]:
        """
        Calls the Gemini API (Vertex, via the new google.genai SDK) with
        explicit handling for every failure mode. Returns (answer_text,
        error_reason, usage) — exactly one of answer_text/error_reason will
        be None.

        `usage` is a dict with keys 'latency_seconds', 'input_tokens',
        'output_tokens', 'thinking_tokens'. Populated whenever a response
        object was actually received from the API — even a safety-blocked
        or otherwise-failed-post-response call still has real latency and
        (usually) real token counts worth capturing. All four values are
        None only when the API call itself raised before any response
        existed (quota exhaustion, malformed request, server error,
        network failure) — there's nothing to measure in that case.

        Token counts read from response.usage_metadata — field names
        (prompt_token_count, candidates_token_count, thoughts_token_count)
        confirmed against Google's own API docs and real SDK response
        objects, not assumed. thinking stays ON for this call (a deliberate
        choice, unlike every other Gemini call site in this project, which
        all set thinking_budget=0) — thoughts_token_count is captured as
        its own field, separate from candidates_token_count, even though
        Google's pricing bills them together as "output" tokens. Keeping
        them separate lets the eventual cost table show how much of a
        Pro call's cost is the visible answer vs. invisible reasoning.

        Failure taxonomy:
          - Safety blocks     → security signal, log at WARNING, return user-safe message
          - Quota exhaustion  → operational signal, log at ERROR
          - Malformed request → developer signal, log at ERROR
          - Unexpected errors → catch-all with full stack trace (exc_info=True)

        Exception types changed with the SDK: the legacy google.api_core
        exceptions (ResourceExhausted, InvalidArgument) don't exist on this
        SDK's error surface. The new SDK raises google.genai.errors.ClientError
        (4xx, carries .code — 429 is quota) and .ServerError (5xx). Catching the
        old exception classes here would compile fine but never actually match
        anything — every real error would silently fall through to the generic
        catch-all with a less specific message.
        """
        empty_usage = {
            "latency_seconds": None,
            "input_tokens": None,
            "output_tokens": None,
            "thinking_tokens": None,
        }
        start = time.perf_counter()
        try:
            response = self.client.models.generate_content(
                model=model_name,
                contents=prompt,
                config=GENERATION_CONFIG,
            )
            latency = time.perf_counter() - start

            usage_meta = getattr(response, "usage_metadata", None)
            if usage_meta is None:
                logger.warning(
                    f"No usage_metadata on response for query '{query[:80]}' — "
                    f"token counts unavailable for this call (latency still captured)."
                )
                usage = {**empty_usage, "latency_seconds": latency}
            else:
                usage = {
                    "latency_seconds": latency,
                    "input_tokens": getattr(usage_meta, "prompt_token_count", None),
                    "output_tokens": getattr(usage_meta, "candidates_token_count", None),
                    "thinking_tokens": getattr(usage_meta, "thoughts_token_count", None),
                }

            text, error_reason = _safe_extract_text(response)

            if error_reason:
                # Safety blocks in a cybersecurity RAG may indicate adversarial
                # content in ingested documents — flag for SOC supervisor review.
                if "BLOCKED" in error_reason or "SAFETY" in error_reason:
                    logger.warning(
                        f"Generation safety-blocked for query '{query[:80]}': "
                        f"{error_reason}. Review ingested content for adversarial material."
                    )
                    return (
                        None,
                        "This query triggered a content safety filter. "
                        "SOC supervisor review recommended.",
                        usage,
                    )
                logger.error(f"Generation failed with reason: {error_reason}")
                return None, "An error occurred during threat analysis.", usage

            logger.info(
                f"Generation successful. Response length: {len(text)} chars. "
                f"Latency: {latency:.2f}s. "
                f"Tokens: in={usage['input_tokens']} out={usage['output_tokens']} "
                f"thinking={usage['thinking_tokens']}."
            )
            return text, None, usage

        except genai_errors.ClientError as e:
            if getattr(e, "code", None) == 429:
                logger.error("Gemini API quota exhausted. Implement exponential backoff.")
                return None, "Service temporarily unavailable — API quota reached.", empty_usage
            logger.error(f"Malformed request to Gemini API: {e}")
            return None, "An error occurred during threat analysis.", empty_usage

        except genai_errors.ServerError as e:
            logger.error(f"Gemini API server error: {e}")
            return None, "Service temporarily unavailable — try again shortly.", empty_usage

        except Exception as e:
            # exc_info=True attaches the full stack trace — saves hours of debugging.
            logger.error(
                f"Unexpected LLM generation error for query '{query[:80]}': {e}",
                exc_info=True
            )
            return None, "An error occurred during threat analysis.", empty_usage

    # ── Public interface ──────────────────────────────────────────────────────

    def generate_answer(
        self,
        query: str | None,
        search_results: dict | None,
        tier: ModelTier | None = None,
    ) -> AnalysisResult:
        """
        Entry point for the generation pipeline.

        FIX 2: Returns AnalysisResult dataclass instead of bare str.
        Callers check result.success to branch UI state and use
        result.source_citations to render ATT&CK technique citations.

        `tier` selects which model this call dispatches to (see _TIER_MODEL).
        None resolves to ModelTier.FLASH — the blind arm (use_routing=False)
        has no routing decision and therefore no tier, and any caller not
        yet tier-aware gets the same model it always got. This keeps the
        blind-arm baseline byte-identical; only routed queries with a real
        tier decision can reach PRO.

        Pipeline stages (fail-fast at each):
          1. Validate query          — input contract
          2. Validate search_results — upstream contract + extract metadatas
          3. Truncate chunks         — chunk-level bounds (keeps metadata in sync)
          4. Build sanitized prompt  — security layer
          5. Call LLM                — generation layer, dispatched by tier
        """
        # Stage 1 — query validation
        query, query_error = self._validate_query(query)
        if query_error:
            return AnalysisResult.fail(query_error)

        logger.info(
            f"Incoming query ({len(query)} chars): "
            f"'{query[:80]}{'...' if len(query) > 80 else ''}'"
        )

        # Stage 2 — search results validation + metadata extraction
        payload, results_error = self._validate_search_results(search_results)
        if results_error:
            return AnalysisResult.fail(results_error)

        context_chunks, metadatas = payload

        logger.info(
            f"RAG SUCCESS: {len(context_chunks)} chunk(s) retrieved. "
            f"Initiating deterministic generation."
        )

        # Stage 3 — chunk-level truncation (keeps chunks + metadatas in sync)
        context_chunks, metadatas = _truncate_chunks(context_chunks, metadatas)

        # Stage 4 — sanitized prompt construction
        prompt = _build_prompt(query, context_chunks)

        # Stage 5 — LLM call, dispatched by tier
        resolved_tier = tier if tier is not None else ModelTier.FLASH
        model_name = _TIER_MODEL[resolved_tier]
        logger.info(f"Dispatching generation: tier={resolved_tier.value} -> model={model_name}")

        answer, llm_error, usage = self._call_llm(prompt, query, model_name)

        if llm_error:
            result = AnalysisResult.fail(llm_error)
        else:
            # Deduplicate citations by technique_id — multiple chunks from the
            # same technique should appear as one citation in the UI, not N.
            seen: set[str] = set()
            unique_citations: list[dict] = []
            for meta in metadatas:
                tid = meta.get("technique_id", meta.get("source", "unknown"))
                if tid not in seen:
                    seen.add(tid)
                    unique_citations.append(meta)

            result = AnalysisResult.ok(answer=answer, citations=unique_citations, context_used=context_chunks)

        # Attached regardless of success/failure — usage is real whenever a
        # response was received (see _call_llm docstring), and a failed
        # generation's latency/token cost is still real spend worth knowing
        # about, even though it won't feed the per-tier cost table (that
        # only aggregates generation_ok=True rows).
        result.latency_seconds = usage["latency_seconds"]
        result.input_tokens = usage["input_tokens"]
        result.output_tokens = usage["output_tokens"]
        result.thinking_tokens = usage["thinking_tokens"]

        return result