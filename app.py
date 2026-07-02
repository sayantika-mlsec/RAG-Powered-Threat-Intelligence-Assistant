"""
app.py — Gradio UI for the RAG-Powered Threat Intelligence Assistant
Entry point for Hugging Face Spaces deployment.

Startup contract:
  - ./brain/         must exist (pre-built ChromaDB, committed to repo)
  - GEMINI_API_KEY   must be set as a HF Space Secret (or in local .env)
  - process_directory() is NOT called here — ingest offline, commit ./brain/

Routing:
  - run_pipeline(query, use_routing=...) is the single entry point called by
    BOTH this UI and the eval harness. use_routing is the only A/B variable:
    False = blind baseline (whole store), True = agentic (route decides corpus).
  - ROUTER_CLIENT uses the new google.genai SDK; ANALYZER still uses the legacy
    google.generativeai SDK. This dual-SDK coexistence is a known, parked
    migration — not an oversight.
"""

import os
import logging
import gradio as gr
from pathlib import Path
from dotenv import load_dotenv

from google import genai
from google.genai import types

from ingest      import ThreatIntelDB
from threat_analyzer  import ThreatAnalyzer, AnalysisResult
from routing     import route_query, Route, _ROUTER_MODEL

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s"
)
logger = logging.getLogger(__name__)

# ── Environment ───────────────────────────────────────────────────────────────
# Loads .env locally; on HF Spaces the Secret is already in the environment.
load_dotenv(dotenv_path=Path(__file__).parent / ".env", override=False)

# ── Constants ─────────────────────────────────────────────────────────────────
DB_PATH    = "./brain"
N_RESULTS  = 5          # chunks retrieved per query — tune against your dataset size
                        # NEW — retrieve more, better chance of hitting the right technique
EXAMPLE_QUERIES = [
    "How do adversaries use phishing for initial access?",
    "What MITRE techniques does Emotet malware use?",
    "How do I detect T1078 Valid Accounts abuse?",
    "What are common lateral movement techniques?",
    "How can defenders detect command and scripting interpreter abuse?",
]

# ── Custom CSS — dark terminal aesthetic ─────────────────────────────────────
# Tone: industrial/utilitarian — this is a SOC tool, not a consumer app.
# Dark background, amber/green accents, monospace data display.
# Deliberately avoids purple gradients and rounded consumer aesthetics.
CUSTOM_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Rajdhani:wght@400;500;600;700&display=swap');

:root {
    --bg-primary:    #0a0d0f;
    --bg-secondary:  #111518;
    --bg-panel:      #141a1f;
    --bg-input:      #0d1117;
    --border-dim:    #1e2d3a;
    --border-active: #2a6496;
    --accent-amber:  #e8a020;
    --accent-green:  #3ddc84;
    --accent-red:    #e05252;
    --accent-blue:   #4fa8d5;
    --text-primary:  #d4dde4;
    --text-dim:      #6b7f8c;
    --text-mono:     #a8c4d4;
    --font-display:  'Rajdhani', sans-serif;
    --font-mono:     'Share Tech Mono', monospace;
}

/* ── Base ──────────────────────────────────────────────────────────────────── */
body, .gradio-container {
    background: var(--bg-primary) !important;
    font-family: var(--font-display) !important;
    color: var(--text-primary) !important;
}

.gradio-container {
    max-width: 960px !important;
    margin: 0 auto !important;
}

/* ── Header ────────────────────────────────────────────────────────────────── */
#header-block {
    border-bottom: 1px solid var(--border-dim);
    padding-bottom: 20px;
    margin-bottom: 8px;
}

#header-block h1 {
    font-family: var(--font-display) !important;
    font-size: 2rem !important;
    font-weight: 700 !important;
    letter-spacing: 0.12em !important;
    color: var(--accent-amber) !important;
    text-transform: uppercase !important;
    margin: 0 !important;
}

#header-block p {
    font-family: var(--font-mono) !important;
    font-size: 0.82rem !important;
    color: var(--text-dim) !important;
    margin: 6px 0 0 0 !important;
    letter-spacing: 0.04em !important;
}

/* ── Input area ────────────────────────────────────────────────────────────── */
#query-input textarea {
    background: var(--bg-input) !important;
    border: 1px solid var(--border-dim) !important;
    border-radius: 4px !important;
    color: var(--text-primary) !important;
    font-family: var(--font-mono) !important;
    font-size: 0.9rem !important;
    padding: 12px 14px !important;
    transition: border-color 0.2s ease !important;
    resize: none !important;
}

#query-input textarea:focus {
    border-color: var(--border-active) !important;
    outline: none !important;
    box-shadow: 0 0 0 2px rgba(42, 100, 150, 0.15) !important;
}

#query-input label {
    font-family: var(--font-mono) !important;
    font-size: 0.75rem !important;
    color: var(--text-dim) !important;
    letter-spacing: 0.08em !important;
    text-transform: uppercase !important;
}

/* ── Submit button ─────────────────────────────────────────────────────────── */
#submit-btn {
    background: var(--accent-amber) !important;
    color: #0a0d0f !important;
    border: none !important;
    border-radius: 3px !important;
    font-family: var(--font-display) !important;
    font-size: 0.95rem !important;
    font-weight: 700 !important;
    letter-spacing: 0.1em !important;
    text-transform: uppercase !important;
    padding: 10px 28px !important;
    cursor: pointer !important;
    transition: background 0.15s ease, transform 0.1s ease !important;
}

#submit-btn:hover {
    background: #f0b030 !important;
    transform: translateY(-1px) !important;
}

#submit-btn:active {
    transform: translateY(0) !important;
}

/* ── Clear button ──────────────────────────────────────────────────────────── */
#clear-btn {
    background: transparent !important;
    color: var(--text-dim) !important;
    border: 1px solid var(--border-dim) !important;
    border-radius: 3px !important;
    font-family: var(--font-display) !important;
    font-size: 0.85rem !important;
    font-weight: 500 !important;
    letter-spacing: 0.06em !important;
    text-transform: uppercase !important;
    transition: border-color 0.15s, color 0.15s !important;
}

#clear-btn:hover {
    border-color: var(--accent-amber) !important;
    color: var(--accent-amber) !important;
}

/* ── Output panels ─────────────────────────────────────────────────────────── */
#answer-output, #citations-output {
    background: var(--bg-panel) !important;
    border: 1px solid var(--border-dim) !important;
    border-radius: 4px !important;
}

#answer-output textarea, #citations-output textarea {
    background: var(--bg-panel) !important;
    color: var(--text-primary) !important;
    font-family: var(--font-mono) !important;
    font-size: 0.88rem !important;
    line-height: 1.65 !important;
    border: none !important;
    padding: 14px !important;
}

#answer-output label, #citations-output label {
    font-family: var(--font-mono) !important;
    font-size: 0.72rem !important;
    color: var(--accent-green) !important;
    letter-spacing: 0.1em !important;
    text-transform: uppercase !important;
    padding: 8px 14px 4px !important;
    border-bottom: 1px solid var(--border-dim) !important;
    display: block !important;
}

/* ── Status bar ────────────────────────────────────────────────────────────── */
#status-bar {
    font-family: var(--font-mono) !important;
    font-size: 0.75rem !important;
    color: var(--text-dim) !important;
    padding: 6px 0 !important;
    letter-spacing: 0.04em !important;
    min-height: 22px !important;
}

/* ── Examples ──────────────────────────────────────────────────────────────── */
.examples-header {
    font-family: var(--font-mono) !important;
    font-size: 0.72rem !important;
    color: var(--text-dim) !important;
    text-transform: uppercase !important;
    letter-spacing: 0.08em !important;
    margin-bottom: 6px !important;
}

.gr-samples-table, .gr-samples-table tr td {
    background: var(--bg-secondary) !important;
    border: 1px solid var(--border-dim) !important;
    color: var(--text-dim) !important;
    font-family: var(--font-mono) !important;
    font-size: 0.82rem !important;
    transition: background 0.15s, color 0.15s !important;
}

.gr-samples-table tr:hover td {
    background: var(--bg-panel) !important;
    color: var(--accent-amber) !important;
    cursor: pointer !important;
}

/* ── Divider ───────────────────────────────────────────────────────────────── */
.section-divider {
    border: none !important;
    border-top: 1px solid var(--border-dim) !important;
    margin: 16px 0 !important;
}

/* ── Footer ────────────────────────────────────────────────────────────────── */
#footer-block p {
    font-family: var(--font-mono) !important;
    font-size: 0.72rem !important;
    color: var(--text-dim) !important;
    text-align: center !important;
    letter-spacing: 0.04em !important;
    margin: 0 !important;
}
"""

# ── System Initialization ─────────────────────────────────────────────────────

def _initialize_systems() -> tuple[ThreatIntelDB | None, ThreatAnalyzer | None, str]:
    """
    Initializes ThreatIntelDB and ThreatAnalyzer at startup.
    Returns (db, analyzer, status_message).

    Both are initialized here — at module load — so Gradio serves the
    first query instantly without a cold-start delay on the first request.

    Returns (None, None, error_msg) on failure so the UI can display a
    meaningful degraded state instead of crashing with a stack trace.
    """
    db = None
    analyzer = None

    # ── DB init ───────────────────────────────────────────────────────────────
    try:
        db = ThreatIntelDB(db_path=DB_PATH)
        chunk_count = db.collection.count()
        if chunk_count == 0:
            logger.warning(
                f"ChromaDB at '{DB_PATH}' is empty. "
                "Commit a pre-built ./brain/ folder to the repo."
            )
            db_status = f"⚠️  DB connected but EMPTY — ingest documents first"
        else:
            logger.info(f"ChromaDB ready: {chunk_count} chunks loaded.")
            db_status = f"DB: {chunk_count} chunks indexed"
    except Exception as e:
        logger.error(f"ChromaDB initialization failed: {e}", exc_info=True)
        return None, None, f"❌ DB init failed: {e}"

    # ── Analyzer init ─────────────────────────────────────────────────────────
    try:
        analyzer = ThreatAnalyzer()
        logger.info("ThreatAnalyzer ready.")
        analyzer_status = "LLM: ready"
    except ValueError as e:
        # Missing API key — common on first HF deploy, surface clearly
        logger.error(f"ThreatAnalyzer init failed: {e}")
        return db, None, f"❌ LLM init failed — check GEMINI_API_KEY Secret: {e}"
    except Exception as e:
        logger.error(f"ThreatAnalyzer init failed: {e}", exc_info=True)
        return db, None, f"❌ LLM init failed: {e}"

    status = f"[ {db_status}  ·  {analyzer_status} ]"
    return db, analyzer, status


def _initialize_router_client() -> genai.Client | None:
    """
    Constructs the google.genai client used by the routing layer.

    Separate from ANALYZER on purpose: the router runs on the new google.genai
    SDK (consistent with routing.py / eval_faithfulness.py) while ThreatAnalyzer
    still runs on the legacy google.generativeai SDK. Both read the same
    GEMINI_API_KEY. Consolidating to one SDK is a tracked, parked migration.

    Returns None on failure (e.g. missing key) so the UI can degrade to the
    blind-retrieval path instead of crashing at module load.
    """
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        logger.warning("GEMINI_API_KEY not set — router unavailable, blind retrieval only.")
        return None
    try:
        client = genai.Client(api_key=api_key)
        logger.info("Router client ready (google.genai).")
        return client
    except Exception as e:
        logger.error(f"Router client init failed: {e}", exc_info=True)
        return None


# Initialize once at module load
DB, ANALYZER, INIT_STATUS = _initialize_systems()
ROUTER_CLIENT = _initialize_router_client()
logger.info(f"Startup status: {INIT_STATUS}")


# ── Routing resolver ──────────────────────────────────────────────────────────

# Sentinel returned by _route_to_corpus for the skip route. Distinct from None
# (which means "no filter, query the whole store") so the two cannot be confused.
SKIP_SENTINEL = "__skip__"


def _route_to_corpus(route: Route) -> str | None:
    """
    Map a Route to the `corpus` argument for semantic_search.

      MITRE_ONLY -> "mitre"        (filter to MITRE chunks)
      KEV_ONLY   -> "kev"          (filter to KEV chunks)
      BOTH       -> None           (no filter — query the whole store)
      SKIP       -> SKIP_SENTINEL  (caller must not retrieve at all)

    Unmapped route raises. A silent default to None (BOTH) would reintroduce
    blind retrieval — the exact behavior routing exists to remove — and would
    quietly corrupt the routing-vs-baseline A/B. Fail loud instead.
    """
    if route == Route.MITRE_ONLY:
        return "mitre"
    if route == Route.KEV_ONLY:
        return "kev"
    if route == Route.BOTH:
        return None
    if route == Route.SKIP:
        return SKIP_SENTINEL
    raise ValueError(f"Unmapped route: {route!r}")


# ── Skip-route no-retrieval response ──────────────────────────────────────────

# System instruction for the skip path. The citation prohibition is load-bearing:
# skip runs with NO retrieved context, so any technique ID or CVE the model emits
# is necessarily ungrounded (hallucinated). Forbidding them keeps the skip path
# from silently bypassing the retrieval-grounding contract the rest of the
# pipeline enforces.
NO_RETRIEVAL_SYSTEM_INSTRUCTION = (
    "You are a threat-intelligence assistant. This query was routed 'skip' — it "
    "needs no knowledge-base lookup (a greeting, a capability/meta question, or "
    "an off-topic message). Answer briefly and directly. Do NOT cite, invent, or "
    "reference any MITRE ATT&CK technique IDs (e.g. T1059) or CVE identifiers — no "
    "threat-intel context was retrieved to support them. If the query actually "
    "needs threat-intel data, say it would need to be looked up rather than "
    "answering from memory."
)


def _no_retrieval_response(query: str) -> AnalysisResult:
    """Skip-route response: a direct LLM answer with no retrieval.

    Returns AnalysisResult with success=True and mode='no_retrieval' — a clean
    success, distinct from a genuine failure. Runs on the router's google.genai
    client and the same model as the router (_ROUTER_MODEL), keeping the skip
    path off the parked legacy SDK and consistent with the routing decision.
    """
    if ROUTER_CLIENT is None:
        # Shouldn't happen — run_pipeline guards ROUTER_CLIENT before routing —
        # but fail loud rather than call .models on None.
        return AnalysisResult.fail("Skip path reached but router client unavailable.")
    try:
        response = ROUTER_CLIENT.models.generate_content(
            model=_ROUTER_MODEL,
            contents=query,
            config=types.GenerateContentConfig(
                system_instruction=NO_RETRIEVAL_SYSTEM_INSTRUCTION,
                temperature=0.0,
                thinking_config=types.ThinkingConfig(thinking_budget=0),
            ),
        )
        text = response.text
        if not text:
            # An empty direct answer is a real failure, not a valid no-retrieval
            # success — do not dress it up as one.
            return AnalysisResult.fail("Skip path returned an empty response.")
        return AnalysisResult.no_retrieval(answer=text)
    except Exception as e:
        logger.error(f"Skip-path generation failed for '{query[:80]}': {e}", exc_info=True)
        return AnalysisResult.fail("Direct (no-retrieval) response generation failed.")


# ── Pipeline ──────────────────────────────────────────────────────────────────

def run_pipeline(
    query: str,
    *,
    use_routing: bool,
) -> tuple[AnalysisResult, dict, str | None]:
    """
    Full RAG pipeline — the single entry point shared by the Gradio UI and the
    eval harness. `use_routing` is the ONLY variable that changes between the
    blind baseline and the agentic version, which is what keeps Friday's A/B
    single-variable:

        use_routing=False -> query the whole store (byte-identical to the
                             pre-routing baseline: precision 0.2389 / recall 0.4667)
        use_routing=True  -> route_query decides the corpus, then retrieval is
                             filtered to it

    Returns (result, search_results, route_value):
        result         : AnalysisResult from the generation layer
        search_results : the raw dict from semantic_search (same shape either
                         path, so downstream citation/status formatting is
                         unchanged)
        route_value    : the route string when routing ran, else None — for
                         logging and eval attribution

    The skip route returns a direct no-retrieval response (mode='no_retrieval'),
    a clean success distinct from a failure. Nothing touches ChromaDB on that
    path.
    """
    empty_results = {"documents": [[]], "metadatas": [[]], "error": None}
    route_value: str | None = None

    if use_routing:
        if ROUTER_CLIENT is None:
            # Router unavailable (e.g. missing key). Do not silently fall back to
            # blind retrieval — that would misattribute the A/B. Fail visibly.
            return (
                AnalysisResult.fail("Router unavailable — check GEMINI_API_KEY."),
                empty_results,
                None,
            )

        decision = route_query(query, ROUTER_CLIENT)
        route_value = decision.route.value
        logger.info(f"Route: {route_value}  ·  reasoning: {decision.reasoning[:120]}")

        corpus = _route_to_corpus(decision.route)

        if corpus == SKIP_SENTINEL:
            # Direct no-retrieval response — a clean success flagged
            # mode='no_retrieval', not a failure. Nothing touches ChromaDB here.
            return (
                _no_retrieval_response(query),
                empty_results,
                route_value,
            )

        search_results = DB.semantic_search(query, n_results=N_RESULTS, corpus=corpus)
    else:
        # Blind baseline — no filter, whole store.
        search_results = DB.semantic_search(query, n_results=N_RESULTS)

    result = ANALYZER.generate_answer(query, search_results)
    return result, search_results, route_value


# ── Core Query Handler ────────────────────────────────────────────────────────

def handle_query(query: str) -> tuple[str, str, str]:
    """
    Gradio event handler — wires the full RAG pipeline to the UI.

    Returns (answer_text, citations_text, status_text) — three outputs
    bound to answer-output, citations-output, and status-bar respectively.

    Never raises — all failures return user-readable strings.
    """
    # ── Input guard ───────────────────────────────────────────────────────────
    if not query or not query.strip():
        return "", "", "⚠  Enter a query above."

    query = query.strip()
    logger.info(f"Query received: '{query[:80]}'")

    # ── System availability check ─────────────────────────────────────────────
    if DB is None:
        return (
            "System unavailable — database failed to initialize.",
            "",
            f"❌ {INIT_STATUS}"
        )

    if ANALYZER is None:
        return (
            "System unavailable — LLM failed to initialize. Check GEMINI_API_KEY.",
            "",
            f"❌ {INIT_STATUS}"
        )

    # ── RAG pipeline (routing on) ─────────────────────────────────────────────
    try:
        result, search_results, route_value = run_pipeline(query, use_routing=True)
    except Exception as e:
        logger.error(f"Unhandled pipeline error: {e}", exc_info=True)
        return (
            "An unexpected error occurred. Please try again.",
            "",
            "❌ pipeline error — check logs"
        )

    # ── Format outputs ────────────────────────────────────────────────────────
    if not result.success:
        route_note = f"  ·  route: {route_value}" if route_value else ""
        return (
            result.error or "Analysis failed.",
            "",
            f"⚠  Query could not be answered — see response above.{route_note}"
        )

    # No-retrieval (skip) success — direct answer, no citations by design.
    if result.mode == "no_retrieval":
        route_note = f"  ·  route: {route_value}" if route_value else ""
        return (
            result.answer,
            "No retrieval performed for this query (no-retrieval mode).",
            f"✓  direct response  ·  no retrieval  ·  "
            f"{len(result.answer)} chars generated{route_note}"
        )

    # Format citations as a clean readable block
    if result.source_citations:
        citation_lines = []
        for i, c in enumerate(result.source_citations, 1):
            technique = c.get("technique_id", "N/A")
            tactic    = c.get("tactic",       "N/A").upper()
            source    = c.get("source",        "N/A")
            date      = c.get("date_added",   "N/A")
            citation_lines.append(
                f"[{i}] {technique}\n"
                f"    TACTIC     : {tactic}\n"
                f"    SOURCE     : {source}\n"
                f"    DATE ADDED : {date}"
            )
        citations_text = "\n\n".join(citation_lines)
    else:
        citations_text = "No ATT&CK technique citations available."

    chunk_count = len(search_results.get("documents", [[]])[0])
    route_note = f"  ·  route: {route_value}" if route_value else ""
    status_text = (
        f"✓  {chunk_count} chunk(s) retrieved  ·  "
        f"{len(result.source_citations)} technique(s) cited  ·  "
        f"{len(result.answer)} chars generated"
        f"{route_note}"
    )

    return result.answer, citations_text, status_text


def handle_clear() -> tuple[str, str, str, str]:
    """Resets all fields to their initial state."""
    return "", "", "", f"[ {INIT_STATUS} ]"


# ── Gradio UI ─────────────────────────────────────────────────────────────────

with gr.Blocks(
    title="Threat Intel Assistant",
    css=CUSTOM_CSS,
    theme=gr.themes.Base(
        primary_hue="orange",
        neutral_hue="slate",
        font=gr.themes.GoogleFont("Rajdhani"),
    )
) as demo:

    # ── Header ────────────────────────────────────────────────────────────────
    with gr.Column(elem_id="header-block"):
        gr.Markdown(
            "# 🛡 THREAT INTEL ASSISTANT\n"
            "RAG-powered analysis · MITRE ATT&CK · Powered by Gemini"
        )

    # ── Query input ───────────────────────────────────────────────────────────
    query_input = gr.Textbox(
        label="QUERY // Ask about a threat, technique, or malware",
        placeholder='e.g. "What MITRE techniques does Emotet use?" or "How is T1566 detected?"',
        lines=3,
        max_lines=6,
        elem_id="query-input"
    )

    with gr.Row():
        submit_btn = gr.Button("⚡ Analyze", variant="primary",  elem_id="submit-btn", scale=2)
        clear_btn  = gr.Button("✕ Clear",   variant="secondary", elem_id="clear-btn",  scale=1)

    # ── Status bar ────────────────────────────────────────────────────────────
    status_bar = gr.Markdown(
        value=f"[ {INIT_STATUS} ]",
        elem_id="status-bar"
    )

    gr.HTML("<hr class='section-divider'>")

    # ── Output panels ─────────────────────────────────────────────────────────
    with gr.Row():
        answer_output = gr.Textbox(
            label="ANALYSIS OUTPUT",
            lines=14,
            max_lines=20,
            interactive=False,
            elem_id="answer-output",
            scale=3
        )
        citations_output = gr.Textbox(
            label="ATT&CK TECHNIQUE CITATIONS",
            lines=14,
            max_lines=20,
            interactive=False,
            elem_id="citations-output",
            scale=2
        )

    gr.HTML("<hr class='section-divider'>")

    # ── Example queries ───────────────────────────────────────────────────────
    gr.Markdown("**EXAMPLE QUERIES**", elem_classes=["examples-header"])
    gr.Examples(
        examples=[[q] for q in EXAMPLE_QUERIES],
        inputs=query_input,
        label=""
    )

    gr.HTML("<hr class='section-divider'>")

    # ── Footer ────────────────────────────────────────────────────────────────
    with gr.Column(elem_id="footer-block"):
        gr.Markdown(
            "Built with ChromaDB · LangChain · Gemini · Gradio  "
            "·  MITRE ATT&CK data  ·  "
            "[GitHub](https://github.com) · [Portfolio](https://huggingface.co)"
        )

    # ── Event wiring ──────────────────────────────────────────────────────────
    submit_btn.click(
        fn=handle_query,
        inputs=[query_input],
        outputs=[answer_output, citations_output, status_bar]
    )

    # Allow Shift+Enter to also submit
    query_input.submit(
        fn=handle_query,
        inputs=[query_input],
        outputs=[answer_output, citations_output, status_bar]
    )

    clear_btn.click(
        fn=handle_clear,
        inputs=[],
        outputs=[query_input, answer_output, citations_output, status_bar]
    )


# ── Entry Point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",   # required for HF Spaces
        server_port=7860,         # HF Spaces default port
        share=False               # set True for a temporary public link locally
    )