"""
app.py — Gradio UI for the RAG-Powered Threat Intelligence Assistant
Entry point for Hugging Face Spaces deployment.

Startup contract:
  - ./brain/         must exist (pre-built ChromaDB, committed to repo)
  - GEMINI_API_KEY   must be set as a HF Space Secret (or in local .env)
  - process_directory() is NOT called here — ingest offline, commit ./brain/
"""

import os
import logging
import gradio as gr
from pathlib import Path
from dotenv import load_dotenv

from ingest      import ThreatIntelDB
from threat_analyzer  import ThreatAnalyzer, AnalysisResult

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


# Initialize once at module load
DB, ANALYZER, INIT_STATUS = _initialize_systems()
logger.info(f"Startup status: {INIT_STATUS}")


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

    # ── RAG pipeline ──────────────────────────────────────────────────────────
    try:
        # Layer 1: Semantic search
        search_results = DB.semantic_search(query, n_results=N_RESULTS)

        # Layer 2: Generation
        result: AnalysisResult = ANALYZER.generate_answer(query, search_results)

    except Exception as e:
        logger.error(f"Unhandled pipeline error: {e}", exc_info=True)
        return (
            "An unexpected error occurred. Please try again.",
            "",
            "❌ pipeline error — check logs"
        )

    # ── Format outputs ────────────────────────────────────────────────────────
    if not result.success:
        return (
            result.error or "Analysis failed.",
            "",
            "⚠  Query could not be answered — see response above."
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
    status_text = (
        f"✓  {chunk_count} chunk(s) retrieved  ·  "
        f"{len(result.source_citations)} technique(s) cited  ·  "
        f"{len(result.answer)} chars generated"
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