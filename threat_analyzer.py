import os
import html
import logging
import google.generativeai as genai
import google.api_core.exceptions
from dotenv import load_dotenv

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ── Environment ───────────────────────────────────────────────────────────────
load_dotenv()

# ── Constants ─────────────────────────────────────────────────────────────────
MAX_QUERY_LENGTH  = 2000   # ~500 tokens — prevents context window abuse
MAX_CONTEXT_CHARS = 12000  # hard ceiling on total ingested chunk size per call

# Model name as env variable — changing models in production = update .env, not code
MODEL_NAME = os.getenv("GEMINI_MODEL_NAME", "gemini-1.5-flash")

SYSTEM_INSTRUCTION = (
    "You are a Senior SOC Analyst Assistant. "
    "Answer ONLY using the threat intelligence provided in the <threat_intelligence> tags. "
    "If the answer cannot be found in that context, respond exactly with: "
    "'I do not have sufficient information in the provided threat intelligence.' "
    "Do not speculate. Do not use prior knowledge. Do not invent IOCs, techniques, or sources."
)

GENERATION_CONFIG = genai.types.GenerationConfig(
    temperature=0.0,
    # top_p and top_k intentionally omitted:
    # at temperature=0.0 greedy decoding is active — sampling parameters
    # have no effect and setting them misleads future maintainers.
    #
    # NOTE: temperature=0.0 produces near-deterministic output but is NOT
    # guaranteed deterministic due to floating point ops across GPU cores.
    max_output_tokens=2048,
    candidate_count=1
)


# ── Module-level prompt utilities ─────────────────────────────────────────────
# These are module-level helpers, NOT defined inside generate_answer().
# Reason: defining functions inside another function redefines them on every
# call, makes them untestable, and signals confused code organisation.

def _sanitize_for_prompt(text: str) -> str:
    """
    Escapes XML/HTML special characters to neutralize prompt injection.
    Converts <, >, & into &lt;, &gt;, &amp; so injected tags are treated
    as plain text content, not prompt structure.

    Critical for a cybersecurity RAG system where ingested content
    (malware reports, phishing samples, threat actor TTPs) is adversarial
    by definition.
    """
    return html.escape(text, quote=True)


def _build_prompt(query: str, context_chunks: list) -> str:
    """
    Constructs the sanitized RAG prompt with hard length limits.
    Both query and all chunks are sanitized before interpolation.
    Truncation warning is logged when context exceeds MAX_CONTEXT_CHARS.
    """
    safe_query  = _sanitize_for_prompt(query.strip())
    safe_chunks = [_sanitize_for_prompt(chunk) for chunk in context_chunks]

    # Join chunks with a visible separator so the model treats them as
    # distinct intelligence reports, not one continuous document.
    context_text = "\n\n---\n\n".join(safe_chunks)

    if len(context_text) > MAX_CONTEXT_CHARS:
        context_text = context_text[:MAX_CONTEXT_CHARS]
        logger.warning(
            f"Context truncated to {MAX_CONTEXT_CHARS} chars "
            f"to stay within token budget."
        )

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
    Safely extracts text from a Gemini response object.
    Returns (text, error_reason) — exactly one will be None.

    Accessing response.text directly raises ValueError when Gemini's
    safety filter blocks a response. This function checks all failure
    modes explicitly before touching .text, and distinguishes:
      - prompt-level blocks  (input was rejected)
      - candidate-level safety filters
      - unexpected finish reasons
      - genuine successful responses
    """
    # Check prompt-level block first (input itself was rejected)
    if response.prompt_feedback.block_reason:
        return None, f"PROMPT_BLOCKED:{response.prompt_feedback.block_reason.name}"

    if not response.candidates:
        return None, "NO_CANDIDATES"

    finish_reason = response.candidates[0].finish_reason.name

    if finish_reason == "SAFETY":
        return None, "CANDIDATE_SAFETY_FILTERED"

    if finish_reason not in ("STOP", "MAX_TOKENS"):
        return None, f"UNEXPECTED_FINISH:{finish_reason}"

    return response.text, None


# ── ThreatAnalyzer class ──────────────────────────────────────────────────────

class ThreatAnalyzer:
    """
    LLM generation layer for the RAG-powered Threat Intelligence Assistant.

    Wrapping in a class (vs module-level initialization) means:
    - ThreatAnalyzer() is instantiated explicitly — testable and mockable
    - self.model can be swapped in tests without patching at module level
    - Model name and config changes are isolated to __init__
    """

    def __init__(self):
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            logger.error("CRITICAL: GEMINI_API_KEY environment variable is missing.")
            raise ValueError("GEMINI_API_KEY not found. Check your .env file.")

        genai.configure(api_key=api_key)

        self.model = genai.GenerativeModel(
            model_name=MODEL_NAME,
            system_instruction=SYSTEM_INSTRUCTION,
            generation_config=GENERATION_CONFIG
        )
        logger.info(f"ThreatAnalyzer initialized with model: {MODEL_NAME}")

    # ── Input validation ──────────────────────────────────────────────────────

    def _validate_query(self, query: str | None) -> tuple[str | None, str | None]:
        """
        Validates and normalises the raw query string.
        Returns (clean_query, error_message) — exactly one will be None.
        """
        if not query or not isinstance(query, str):
            logger.warning("generate_answer called with empty or non-string query.")
            return None, "Invalid query. Please provide a non-empty search string."

        query = query.strip()

        if len(query) < 3:
            logger.warning(f"Query too short to be meaningful: '{query}'")
            return None, "Query too short. Please provide more detail."

        if len(query) > MAX_QUERY_LENGTH:
            logger.warning(
                f"Query exceeds {MAX_QUERY_LENGTH} chars ({len(query)}). Truncating."
            )
            query = query[:MAX_QUERY_LENGTH]
            # Truncation, not rejection — a 2001-char query is probably
            # valid. Hard rejection creates unnecessary friction for
            # legitimate SOC analysts under pressure.

        return query, None

    def _validate_search_results(
        self, search_results: dict | None
    ) -> tuple[list | None, str | None]:
        """
        Validates the search_results structure returned by semantic_search().
        Returns (context_chunks, error_message) — exactly one will be None.

        Three distinct failure modes are distinguished:
          1. None  — upstream semantic_search() failed entirely
          2. Empty — search succeeded but found no relevant chunks
          3. Malformed — unexpected structure from ChromaDB API change
        """
        # Guard 1: None check — semantic_search() returns None on exception
        if search_results is None:
            logger.warning(
                "generate_answer received None search_results. "
                "Upstream semantic_search() likely failed."
            )
            return None, "Threat intelligence search failed. Cannot generate analysis."

        # Guard 2: Safe structural extraction
        try:
            documents     = search_results.get('documents', [])
            context_chunks = documents[0] if documents else []
        except (IndexError, TypeError, AttributeError) as e:
            logger.error(
                f"Malformed search_results structure: {e}. "
                f"Got type: {type(search_results)}"
            )
            return None, "Threat intelligence search returned malformed data."

        # Guard 3: Empty context — triage gate, do NOT call the LLM
        if not context_chunks:
            logger.warning("RAG BYPASS: No relevant chunks found. Aborting LLM call.")
            return None, "No relevant threat intel found in the database. I cannot answer this query."

        return context_chunks, None

    # ── Core generation ───────────────────────────────────────────────────────

    def _call_llm(self, prompt: str, query: str) -> str:
        """
        Calls the Gemini API with explicit handling for every failure mode:
          - Safety blocks (security signal — log at WARNING)
          - Quota exhaustion (operational signal — log at ERROR)
          - Malformed requests (developer signal — log at ERROR)
          - Unexpected errors (catch-all with full stack trace)
        """
        try:
            response = self.model.generate_content(prompt)
            text, error_reason = _safe_extract_text(response)

            if error_reason:
                # Safety blocks are a security signal in a cybersecurity RAG:
                # they may indicate adversarial content in ingested documents.
                if "BLOCKED" in error_reason or "SAFETY" in error_reason:
                    logger.warning(
                        f"Generation safety-blocked for query '{query[:80]}': "
                        f"{error_reason}. Review ingested content for adversarial material."
                    )
                    return (
                        "This query triggered a content safety filter. "
                        "SOC supervisor review recommended."
                    )
                logger.error(f"Generation failed with reason: {error_reason}")
                return "An error occurred during threat analysis."

            logger.info(
                f"Generation successful. Response length: {len(text)} chars."
            )
            return text

        except google.api_core.exceptions.ResourceExhausted:
            logger.error(
                "Gemini API quota exhausted. Implement exponential backoff."
            )
            return "Service temporarily unavailable — API quota reached."

        except google.api_core.exceptions.InvalidArgument as e:
            logger.error(f"Malformed request to Gemini API: {e}")
            return "An error occurred during threat analysis."

        except Exception as e:
            # exc_info=True attaches the full stack trace to the log entry.
            # In production, stack traces in logs save hours of debugging.
            logger.error(
                f"Unexpected LLM generation error for query '{query[:80]}': {e}",
                exc_info=True
            )
            return "An error occurred during threat analysis."

    # ── Public interface ──────────────────────────────────────────────────────

    def generate_answer(self, query: str | None, search_results: dict | None) -> str:
        """
        Entry point for the generation pipeline.

        Pipeline order:
          1. Validate query  (input contract)
          2. Validate search_results  (upstream contract)
          3. Build sanitized prompt  (security layer)
          4. Call LLM  (generation layer)

        Each stage fails fast with a specific, logged message.
        No stage swallows errors silently.
        """
        # Stage 1 — query validation
        query, query_error = self._validate_query(query)
        if query_error:
            return query_error

        logger.info(
            f"Incoming query ({len(query)} chars): "
            f"'{query[:80]}{'...' if len(query) > 80 else ''}'"
        )

        # Stage 2 — search results validation
        context_chunks, results_error = self._validate_search_results(search_results)
        if results_error:
            return results_error

        logger.info(
            f"RAG SUCCESS: {len(context_chunks)} chunks retrieved. "
            f"Initiating deterministic generation."
        )

        # Stage 3 — sanitized prompt construction
        prompt = _build_prompt(query, context_chunks)

        # Stage 4 — LLM call with explicit failure handling
        return self._call_llm(prompt, query)