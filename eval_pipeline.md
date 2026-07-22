# RAG Evaluation Pipeline — Current State

*Last updated: July 22, 2026. This document reflects the current, validated state of the Tiered RAG evaluation pipeline. Full build history, dated findings, and superseded/retracted claims live in `docs/lab-notes.md` — this file states where things stand, not how they got there.*

---

## Terminology Note

The word "gate" was previously overloaded across three unrelated concepts in this pipeline. As of this revision:

| Term | Meaning | Legacy name(s) |
|---|---|---|
| **Skip Rows** | Queries with no expected IDs at all (technique or CVE) — never retrieval-scored. q014, q017–q020. | `n_gated_out` (code field name retained, not renamed) |
| **Eligibility Filter** | The recall@3 > 0 requirement a row must pass to be faithfulness-scored. Rows failing this are faithfulness-ineligible, not skipped. | "gated out of faithfulness scoring" |
| **Confidence Gate** | The dense-search-vs-rewrite decision in `retrieve_for_route()`, based on whether the top-1 distance is at or below a calibrated threshold. | "the gate" (ambiguous with the above two) |

Code field names (`n_gated_out`, etc.) are unchanged; this table exists so prose usage is unambiguous going forward, particularly as next project's instrumentation wrapper begins logging fields with similar names.

---

## Why Evaluate At All

A RAG demo answers questions. A RAG *system* has to prove its answers are trustworthy under questioning — which retrieved chunks were right, which answers were actually grounded in them, and whether the user actually got the right answer. The evaluation splits into three independently meaningful axes:

- **Retrieval** — did the right chunks come back in top K?
- **Faithfulness** — given those chunks, is the answer actually grounded in them?
- **Correctness** — regardless of grounding, does the answer convey the same facts as the verified gold answer?

Faithfulness and correctness are deliberately orthogonal, not redundant. A refusal can be perfectly faithful (no ungrounded claims) and still completely incorrect (user got nothing). An answer can be correct by a lucky ungrounded guess. Reading the two together turns one blurry number into a diagnostic: a failure on faithfulness alone points at generation discipline; a failure on correctness alone (with faithfulness intact) points at retrieval or corpus completeness; a failure on both is a real double failure.

## Eval-Set Construction

20 queries (`eval_set.json`), IDs q001–q020, each with `expected_technique_ids` and/or `expected_cve_ids`, plus a hand-written gold-answer summary. Gold answers are hand-written and verified against ingested chunks — an AI-generated gold would only measure agreement between two models, not correctness.

- **15 scored rows (Group A)** — the reconciliation total used throughout, and the exact set both faithfulness and correctness score against.
- **5 Skip Rows** (q014, q017–q020) — no expected IDs of any kind, never retrieval-scored.
- **q011, q012** — empty `expected_technique_ids` but scored against `expected_cve_ids`; not Skip Rows.

## Metrics

**Precision@3 / Recall@3** — measured against `expected_technique_ids ∪ expected_cve_ids`. Precision's denominator is the actual retrieved count, not a fixed 3 (some paths return fewer). Recall@3 = 0 makes a row Eligibility-Filter-excluded from faithfulness scoring — there's no correct context to be faithful to.

**Faithfulness** — Gemini 2.5 Flash as judge, 1–5 rubric, judge sees only the answer and context (not the gold). Measures grounding, not correctness — an answer can be faithful and still wrong, or refuse and score high for making no claims to contradict (see Known Limitations).

**Faithfulness scores are directional, not precisely comparable across arms.** Repeated observation (q013 flipping 1→5 on identical retrieval and identical refusal behavior — judge variance on an unchanged input) shows single-run judge scores carry enough variance that a 0.5–1.0 point delta between arms should be read as noise unless independently corroborated. Means are reported below for completeness but should not be treated as precise deltas.

**Correctness-vs-gold** (`correctness_score.py`, built July 22, 2026) — a separate Gemini-as-judge scorer, 1–5 rubric, comparing the candidate answer against the hand-written gold answer (`expected_answer_summary`) only — **never** sees retrieved context. Deliberately orthogonal to faithfulness: an ungrounded lucky guess that happens to match gold still scores well here; that's a faithfulness problem, not a correctness one. Unlike faithfulness, correctness has **no eligibility gate** — all 15 Group A rows are scored unconditionally, every run, because a refusal caused by bad retrieval is exactly the case this metric needs to capture, not exclude. A refusal naturally falls to the rubric's bottom band ("no correct information delivered... or is a refusal/non-answer") without any code-level special case.

---

## Current Numbers (as of July 22, 2026 — post-fragmentation-fix, post-correctness-metric)

### Retrieval — current architecture (merge + guaranteed-slot rerank + fragmentation resolution)

| Arm | Precision@3 | Recall@3 |
|---|---|---|
| Blind | 0.2444 | 0.5667 |
| Routed, no fixes | 0.2444 | 0.5667 |
| Routed + fixes (merge + guaranteed-slot rerank, pre-fragmentation-fix) | 0.5111 | 0.9444 |
| **Routed + fixes + fragmentation resolution (current)** | **0.5333** | **0.9778** |

Five rows now confirmed at 1.00 recall: q003, q010, q013, q016 (unchanged from before), plus **q015 (0.50 → 1.00)** — the fragmentation fix's one confirmed real fix among the targeted near-ties. q005 unchanged at 0.67 — targeted-recheck confirmed fragmentation is **not** the cause for this row (see Known Limitation #7). No row regressed.

**Comparison against the pre-rearchitecture (confidence-gated) reference point** — a different pipeline entirely, kept for context, not a like-for-like baseline:

| | Recall | Precision |
|---|---|---|
| Old gated arm (retired architecture) | 0.8222 | 0.4778 |
| Current architecture | **0.9778** | **0.5333** |

Beats the old architecture on both axes simultaneously, same as the pre-fragmentation-fix architecture already did.

### Faithfulness — RE-SCORED under the current architecture, July 22, 2026

| Arm | Mean | N Eligible | N Ineligible |
|---|---|---|---|
| Blind | 4.444 | 9 | 6 |
| Routed (pre-fix retrieval, stale) | 5.000 | 9 | 6 |
| Routed + fixes + Confidence Gate (retired architecture) | 4.429 | 14 | 1 |
| **Routed + fixes + fragmentation resolution (current)** | **4.0** | **15** | **0** |

**Full eligibility for the first time — 15/0, not 14/1.** q004, the sole gated-out row under every prior architecture, now retrieves successfully and scores a clean 5.

**4.0 is not directly comparable to 4.429 as a clean delta.** Different N, materially different underlying retrieval, and two confirmed judge-calibration errors (Known Limitations #5 and #8) rather than genuine faithfulness failures. The pipeline's real faithfulness is plausibly higher than 4.0 once those are accounted for — a judging-quality question to name, not a number to quietly adjust.

**q015's initial score (2/5) was a `MAX_TOKENS` truncation bug, not a finding.** `threat_analyzer.py`'s `GENERATION_CONFIG` capped `max_output_tokens` at 2048, shared between thinking and output tokens with no separate budget — q015's compound cross-corpus answer used 1,966 thinking + 78 output = 2,044/2,048, four tokens from the wall. `_safe_extract_text()` also treated `MAX_TOKENS` as equivalent to `STOP`, so the truncated answer returned as a clean, unflagged success. Fixed: cap raised to 4096; `MAX_TOKENS` now logs a warning instead of passing silently. Re-verified: 93 output + 1,248 thinking = 1,341 tokens, real headroom; answer complete, ends on a full sentence; score moved 2 → 4. The table above reflects the post-fix number.

### Correctness-vs-Gold — first run, July 22, 2026

| Metric | Value |
|---|---|
| Correctness mean | **3.533** |
| N scored | **15 / 15** (no eligibility gate — see Metrics) |

First-ever run of this metric — this number **is** the baseline, nothing to diff against yet.

**Reading faithfulness and correctness together, per row, is where this metric earns its keep:**

| id | Faithfulness | Correctness | Read |
|---|---|---|---|
| q006 | 1 | 3 | independent cross-metric corroboration that faithfulness's 1 is miscalibrated, not a real defect — see Known Limitation #8 |
| q009 | 1 | 1 | agree, but for different reasons — correctness's 1 is definitionally correct (refusal delivers none of gold's facts); faithfulness's 1 remains the miscalibrated one (context-verified, see Known Limitation #5) |
| q011, q012 | 4 | 3 | faithfulness passes an honest hedge; correctness catches that the hedge omits gold's "why it's on KEV" content — see Known Limitation #9 |
| q015, q016 | 4, 3 | 3, 3 | same KEV-omission pattern as q011/q012 — see Known Limitation #9 |
| q010 | 5 | 3 | correctness docked a correct, complete answer against a deliberately terse gold — see Known Limitation #10, unconfirmed |

### Tier Distribution & Cost/Latency (Routed Arm)

13 flash / 7 pro (35% pro), captured under the pre-rearchitecture pipeline. Not re-verified post-rearchitecture or post-fragmentation-fix — still open (see Deferred Work).

---

## Architecture (Current Pipeline — Fourth Revision, July 21, 2026)

1. **Route** — `route_query()` picks a corpus (`mitre_only` / `kev_only` / `both` / `skip`) and a model tier (`flash` / `pro`) in one structured-output call. `temperature=0.0` pinned.

2. **Retrieve** — `retrieve_for_route()`:
   - **Exact match** — literal CVE/technique IDs in the query fetched directly via ChromaDB metadata filter. Deterministic.
   - **Otherwise:** dense search on the raw query (widened to `config.RERANK_POOL_K=15`) **and** rewrite/decomposition (`retrieve_with_rewrite()`, also widened) **both run unconditionally** — no gate.
   - **Fragmentation resolution** — before dedup/rerank runs, `_resolve_fragments()` replaces any candidate whose `technique_id` is in the live-computed fragmented set with one candidate built from **all** its chunks, fetched fresh via `db.collection.get()`. `_fragmented_ids(db)` computes the fragmented set once per process from live collection metadata. Only the ~9.8% of technique_ids that are actually fragmented trigger the extra DB call.
   - **Guaranteed slot per sub-query** — each sub-query's own best (post-resolution) candidate, reranked against **its own sub-query text**, gets a seat first.
   - **Fill** — remaining slots, reranked against the **original raw query**.
   - Cross-encoder scoring (`reranker.py`, `cross-encoder/ms-marco-MiniLM-L-12-v2`) strips structural metadata headers and markdown cross-reference links before scoring — **never** from what's returned, cited, or generated.
   - **Confidence gate: REMOVED ENTIRELY.**

3. **Generate** — `ThreatAnalyzer.generate_answer()` dispatches to the tier-selected model, strict grounding instruction (context-only, exact refusal string on any gap).

4. **Score** — retrieval scored by precision/recall@3; faithfulness scored by LLM-judge against retrieved context (not gold), gated on eligibility; **correctness scored separately (`correctness_score.py`) by LLM-judge against gold (not context), all 15 Group A rows, no gate.** All runs logged to MLflow with fail-loud reconciliation invariants and lineage (`recall_run_id` for faithfulness; `eval_set_path` + `capture_artifact` for correctness).

---

## Known Limitations

**1. Chunk fragmentation at ingestion — FIXED (scoring-level), July 21, 2026.**

Full-corpus sweep: 179/1823 (9.8%) technique/CVE entries fragmented across multiple chunks; max T1034 at 7 chunks. **Fix shipped:** `_resolve_fragments()` concatenates a fragmented entry's chunks (fetched fresh) before dedup/rerank ever sees them. Verified: q003 confirmed genuinely fixed (5.3-point margin on complete content, not a coin flip); q015 confirmed fixed, fragmentation confirmed as actual root cause; q005 confirmed **not** fixed by this change (see Limitation #7). Known residual: `_fragmented_ids()`'s cache is process-lifetime, accepted tradeoff; the other ~176 fragmented entries beyond the three targeted have not been individually checked for retrieval impact.

**2, 3, 4. RESOLVED BY REMOVAL.** The confidence gate's threshold-precision-loss, unreconciled overlap-zone count, and resubstitution-not-held-out-validation limitations are retired along with the gate itself (July 21, 2026). Full rationale in `docs/lab-notes.md`. Not re-numbered, to avoid breaking cross-references to #5+.

**5. Faithfulness judge is lenient toward wrongful refusal — bidirectional, both directions now confirmed stable.**

An answer with no claims trivially passes the grounding check. Confirmed on q013 and q015 (blind vs. routed: same refusal, scores 1 then 5).

**q009 — resolved via k=3 replication, July 22, 2026.** q009 scored 1 across all three independent judge calls (identical score, identical reasoning each time) — stable judge behavior, not noise in the q013 sense. Manual verification of the retrieved context (`generation_capture_post_frag.json`) confirms the refusal was correct: the T1047 chunk fully describes WMI's abuse for execution, but the phrase "Web-Based Enterprise Management" / WBEM never appears in any of the three retrieved chunks. The query asks the model to confirm a real-world fact the corpus text never states — asserting it would violate the system's own no-outside-knowledge grounding contract. The judge's reasoning ("the context contains detailed information about multiple execution techniques") checks topical relevance, not whether the *specific* claim being asked about is supported — that's the actual miscalibration.

This confirms the July 18 diagnosis (legitimate refusal, eval-set defect) over the judge's score, and confirms this limitation is bidirectional (leniency toward some refusals per q013/q015, over-penalization of others per q009), not one-directional as the q003 retraction had concluded.

**Decision, July 22, 2026: left as-is.** Not patching the judge prompt (would dilute faithfulness's narrow "grounding only" contract for one query) and not revising q009's phrasing (a legitimate test of corpus-completeness handling). Documented as a known, stable judge-calibration gap. **Independently corroborated by the correctness metric** (see Current Numbers, above): correctness's own score of 1 on q009 is definitionally correct (a refusal delivers none of gold's facts) and requires no such caveat — the two metrics agree on the number for different, non-conflicting reasons.

**6. Tier classification instability, q003 (gated arm), unresolved.** Unchanged, still open. Distinct subsystem from the *separate* q003 retrieval-near-tie finding (Limitation #7's build history) — routing-layer tier classification vs. retrieval-layer reranking, diagnosed in different sessions, kept separate deliberately.

**7. Reranker cannot reliably discriminate between genuinely similar techniques — REVISED July 21, 2026, post-fragmentation-fix.**

**q015 (T1203 vs. T1190) — resolved.** Post-fix, T1190 pulls 8.5 points clear (9.5639 vs. 1.1060) — a fragmentation artifact, now closed.

**q005 (T1621 vs. T1111) — NOT resolved by the fragmentation fix.** Both fully resolved on complete text; the gap **widened** (1.39 → 1.65 points). Real cause: genuinely subtle technique-vs-technique distinction, plus guaranteed-slot budget exhaustion (q005's 3 sub-queries exactly fill `RETRIEVAL_TOP_K=3`, so no fill step ever runs). Both remain open and unfixed. Candidate fixes (`bge-reranker-base` trial, top-2-per-subquery guarantee) not pursued without stronger evidence than one query.

**8. Faithfulness judge over-penalizes trivial framing echoed from the query itself — confirmed stable via k=3 replication, July 22, 2026, and independently corroborated by the correctness metric.**

q006's answer states "The parent technique is called Process Injection" — T1055's chunk fully supports the substantive claim word-for-word. The word "parent" is an echo of the query's own phrasing, not an asserted taxonomy fact. Scored 1 ("largely fabricated") across all three independent replication calls (identical score, identical reasoning each time) — contradicting the rubric's own worked example, which scores trivial connective/framing language a 4, not a 1.

**Independently corroborated, July 22, 2026:** the correctness judge — a different judge, different rubric, same answer — scored q006 a 3, explicitly crediting "correctly identifies the parent technique as Process Injection" and only docking missing supporting detail, not fabrication. Two independent judges converge on "the answer is substantively right"; only faithfulness's rubric application is wrong.

**Not patched this session** — same scope-holding rationale as #5: a rubric change on one confirmed query risks overcorrecting for a narrow case. Left as documented, deferred limitation.

**9. KEV corpus chunks never state "actively exploited" as literal text — new, July 22, 2026, confirmed on all affected rows.**

Direct inspection of all 10 retrieved KEV chunks (across q011, q012, q013, q015, q016) confirms: the phrase "actively exploited" does not appear in any KEV chunk's text. Exploitation status is implicit in a CVE's *membership* in the KEV catalog, but never asserted as a sentence in the ingested chunk. Result: any query phrasing that asks the model to explicitly confirm "is this actively exploited" produces an honest, correctly-grounded hedge — confirmed on q011, q012, q015, q016 (each shows the same explicit "I do not have sufficient information... to explain why it's on CISA's known-exploited list" or equivalent). q013 avoids the pattern only because its phrasing ("which flaw... is in the exploited catalog") doesn't force an explicit exploitation-status assertion.

**Distinct in kind from #5 and #8** — not a judge-rubric miscalibration on either side. This is a real gap in what the corpus text asserts, surfaced by the same eval-set/corpus mismatch shape as q009 (Limitation #5), but affecting four rows via a shared, identified root cause rather than one query's specific phrasing.

**Confirms the value of the two-metric framework:** faithfulness does not catch this (the hedge is honestly grounded — scores 4 on q011/q012). Correctness does (gold assumes KEV membership implies confirmed exploitation — scores 3 on all four affected rows). First confirmed case this project has of the correctness metric surfacing a real gap neither retrieval scoring nor faithfulness alone would show.

**Not fixed this session.** Two paths, neither pursued yet: corpus enrichment (add an explicit exploitation-status line at ingestion — a real content change, out of scope for a scoring-level fix) or accept the model's hedge as correct behavior and treat gold's assumption as the actual defect (same resolution class as q009). Documented as understood, deferred — see Deferred Work.

**10. Correctness judge may under-credit a complete answer against a deliberately terse gold — new, July 22, 2026, single instance, NOT confirmed systematic.**

q010's `expected_answer_summary` deliberately withholds the technique's name (`difficulty_note`: tests definition-matching *without* keyword hints — the gold is intentionally terse by eval-set design). The candidate answer correctly named T1105. Correctness scored this a 3, reasoning "the gold answer describes the technique without naming it" — docking the candidate for including *more* correct information than a deliberately incomplete gold.

Possible mirror image of Limitation #8 (a judge over-penalizing a superset of correct content relative to what it's being compared against), but in the correctness judge rather than faithfulness. **Single instance — unlike #8/#9, not replicated.** Whether this is stable miscalibration or a one-off has not been tested. k=3 replication (same protocol used for q006/q009) would resolve it; not run this session — explicitly deferred, not dismissed.

---

## Deferred Work (not scheduled, not forgotten)

- Fragmentation fix (Limitation #1) — **done, July 21, 2026.**
- q005's guaranteed-slot-exhaustion + reranker-discrimination limitation (Limitation #7, q005 half) — still open, precisely diagnosed. Candidate fixes: `bge-reranker-base` trial, top-2-per-subquery guarantee. Neither pursued without stronger evidence than one query.
- Re-score faithfulness under the current architecture — **done, July 22, 2026.** Mean 4.0, 15/15 eligible.
- Correctness-vs-gold metric — **done, July 22, 2026.** `correctness_score.py` built and run: mean 3.533, N=15/15. First baseline recorded above.
- Root-cause q003's gated-arm tier instability (Limitation #6) — arm no longer exists post-rearchitecture; whether the finding still applies is itself open.
- Judge k=3 replication on q006 and q009 — **done, July 22, 2026.** Both confirmed stable (3/3 identical), not noise. q013/q015's original k=3 replication remains separately unexecuted.
- **New: k=3 replication on q010's correctness score (Limitation #10)** — not run this session. Needed before treating it as a real correctness-judge miscalibration rather than a one-off.
- **New: KEV corpus-completeness fix or eval-set revision (Limitation #9)** — two candidate resolutions named, neither pursued. Deferred beyond the hardening-phase close.
- Tier distribution / cost-latency re-verification under the current architecture — still stale, unrelated to this session's threads.
- Eval-set growth past 20 queries — repeatedly named as the resolution to several small-n caveats, not yet scheduled.
- Hybrid BM25 — deprioritized; q008/q010's original vocabulary-mismatch framing turned out not to require it. Still open for any query where a genuine coverage gap (not ranking) is confirmed.

## See Also

Full chronological build log, dated findings, retracted claims, and the reasoning behind each fix: `docs/lab-notes.md`.