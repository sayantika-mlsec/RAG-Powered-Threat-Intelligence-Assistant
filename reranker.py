"""
reranker.py — Cross-encoder reranking for merged retrieval candidate pools.

Loaded lazily (module-level singleton, populated on first real use) — same
lazy-load spirit as ThreatIntelDB's embedding function, but deferred past
import time so importing this module never pays the ~80MB model-load cost
unless reranking actually runs (e.g. a unit test that only exercises
exact_id.py shouldn't pull this model in).

Model: set via config.RERANK_MODEL — MS MARCO passage-ranking family
(ms-marco-MiniLM-L-6-v2 / L-12-v2) as of writing. Local, no API cost.
CPU-fine for the pool sizes used here (<=30 candidates per query, per
config.RERANK_POOL_K).

SCORE DIRECTION — READ THIS BEFORE WIRING INTO ANYTHING:
Higher = more relevant. This is the OPPOSITE of ChromaDB's cosine distance
(lower = more similar). Any caller that populates a "distances"-shaped field
with this module's scores must not assume the old lower-is-better ordering.
"""

import logging
import re
from sentence_transformers import CrossEncoder

import config

logger = logging.getLogger(__name__)

_model: CrossEncoder | None = None

# Structural header lines ingest.py embeds VERBATIM into chunk text
# (TECHNIQUE_ID:, TACTIC:, etc. are real content to the chunker, not
# stripped metadata). TWO DIFFERENT KINDS OF FIELD, handled differently —
# conflating them was a real bug, corrected 2026-07-21 (see below):
#
#   ADMINISTRATIVE/CATEGORICAL — TECHNIQUE_ID, VULNERABILITY_ID, TACTIC,
#   DATE_ADDED, Platforms, Vendor, Product, Patch Due Date. Low information
#   content, shared across many unrelated entries (TACTIC: Initial Access
#   applies to dozens of techniques). Confirmed confound on q010: a query
#   opening "Once initial access is achieved..." scored a chunk containing
#   the literal line "TACTIC: Initial Access" above every genuinely
#   relevant candidate. FULL LINE stripped, value included — there's
#   nothing worth keeping.
#
#   DESCRIPTIVE TITLE — Technique Name, Vulnerability Name. Each VALUE is
#   close to a one-line summary of what the chunk is actually about —
#   legitimate, high-signal content, not noise. Originally stripped
#   entirely alongside the administrative fields (Jul 21, first pass) —
#   that was a real regression, confirmed on q013: CVE-2017-0005's
#   Vulnerability Name is "...GDI Privilege Escalation Vulnerability", and
#   with that line gone, its body text only says "gain privileges" (not
#   the literal query phrase "privilege escalation"), while a WRONG
#   candidate (CVE-2014-1812) happened to restate "privilege escalation"
#   verbatim in its own body and won purely on that coincidence. Only the
#   LABEL ("Technique Name:") is stripped now — the value stays, so
#   "GDI Privilege Escalation Vulnerability" remains scoreable content.
#
# Field list is based on header patterns OBSERVED in real retrieved chunk
# text this session (both MITRE and KEV formats), not a repo-wide sweep of
# threat_reports/*.txt — widen if a future diagnostic run surfaces a field
# not covered here.
_HEADER_FULL_LINE_PATTERN = re.compile(
    r"^\s*(TECHNIQUE_ID|VULNERABILITY_ID|TACTIC|DATE_ADDED|Platforms|Vendor|"
    r"Product|Patch Due Date)\s*:.*$",
    re.IGNORECASE | re.MULTILINE,
)
_HEADER_LABEL_ONLY_PATTERN = re.compile(
    r"^\s*(Technique Name|Vulnerability Name)\s*:\s*",
    re.IGNORECASE | re.MULTILINE,
)

# MITRE ATT&CK technique write-ups routinely name-drop related techniques
# as markdown links — "(i.e., [Ingress Tool Transfer](.../T1105))",
# "Unlike [Exploit Public-Facing Application](.../T1190), ...". Confirmed
# TWICE, independently, 2026-07-21:
#   - T1570's chunk names T1105 in a loose "(i.e., ...)" aside — T1570
#     (wrong answer) outscored T1105 (correct) on q010.
#   - T1189's chunk names T1190 inside an explicit NEGATION — "Unlike
#     [T1190], this technique...". T1189 (wrong answer) STILL outscored
#     T1190 (correct) on q015, despite the sentence saying the opposite of
#     what the model rewarded it for.
# The second case is why this strips UNIFORMLY rather than trying to keep
# links that appear in "legitimate" sentence positions: if the model can't
# use an explicit "Unlike X" to avoid rewarding X, surrounding grammar
# can't be trusted as a signal for which links are safe to leave in.
# Scope: /techniques/ links only (parent and sub-technique, e.g. T1027 and
# T1027/010) — this is what's CONFIRMED. Links to /software/, /groups/,
# /tactics/ are NOT stripped; not confirmed as a problem, not assumed to
# be one either.
_TECHNIQUE_CROSSREF_PATTERN = re.compile(
    r"\[[^\]]+\]\(https://attack\.mitre\.org/techniques/[A-Za-z0-9./]+\)"
)


def _strip_metadata_header(text: str) -> str:
    """
    Removes ADMINISTRATIVE/CATEGORICAL header lines ENTIRELY, and removes
    only the LABEL (not the value) from Technique Name:/Vulnerability
    Name: lines — see the field-list comment above for why these two
    categories are treated differently. Scoring-only: callers of rerank()
    still receive the ORIGINAL, untouched document text in the returned
    tuples — this function's output never leaves the pair-construction
    step inside rerank().
    """
    text = _HEADER_FULL_LINE_PATTERN.sub("", text)
    text = _HEADER_LABEL_ONLY_PATTERN.sub("", text)
    return text.strip()


def _strip_technique_crossrefs(text: str) -> str:
    """
    Removes markdown links naming OTHER techniques (both the visible
    bracketed name AND the URL — removing just the URL would leave the
    canonical technique name sitting there as plain text, which is the
    literal string that caused both confirmed failures). Scoring-only,
    same isolation contract as _strip_metadata_header: never affects what
    rerank() returns, cites, or generates from.

    Cleans up double-spaces left by removal (e.g. "Unlike , the focus")
    for scoring hygiene — cosmetic, not semantic; the model doesn't need
    grammatically perfect input, just accurate content.
    """
    stripped = _TECHNIQUE_CROSSREF_PATTERN.sub("", text)
    return re.sub(r"[ \t]{2,}", " ", stripped).strip()


def _get_model() -> CrossEncoder:
    global _model
    if _model is None:
        logger.info(f"Loading cross-encoder: {config.RERANK_MODEL}")
        _model = CrossEncoder(config.RERANK_MODEL)
    return _model


def rerank(
    query: str,
    candidates: list[tuple[dict, str]],
    top_k: int,
) -> list[tuple[dict, str, float]]:
    """
    Scores each (metadata, document_text) candidate against `query` with a
    cross-encoder. Returns the top_k, sorted DESCENDING by score.

    The TEXT SCORED has structural metadata headers AND markdown technique
    cross-reference links stripped (see _strip_metadata_header and
    _strip_technique_crossrefs) — the text RETURNED in the output tuples
    is the original, untouched document_text from `candidates`. Downstream
    consumers (citations, generation) never see the stripped version.

    candidates: already-deduped (metadata, document_text) pairs — this
    function does not dedupe, only scores and ranks what it's given.

    Returns length min(top_k, len(candidates)) — NOT padded. Same principle
    as exact-match's variable-length return: a genuinely smaller candidate
    set should report as smaller, not be padded to look uniform.
    """
    if not candidates:
        return []

    model = _get_model()
    pairs = [
        (query, _strip_technique_crossrefs(_strip_metadata_header(doc_text)))
        for _, doc_text in candidates
    ]
    scores = model.predict(pairs)

    scored = list(zip(scores, candidates))  # original (meta, doc) — unstripped
    scored.sort(key=lambda x: x[0], reverse=True)  # higher score = more relevant

    return [(meta, doc, float(score)) for score, (meta, doc) in scored[:top_k]]