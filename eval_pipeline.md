# RAG Evaluation Pipeline — Current State

*Last updated: July 21, 2026. This document reflects the current, validated state of the Tiered RAG evaluation pipeline. Full build history, dated findings, and superseded/retracted claims live in `docs/lab-notes.md` — this file states where things stand, not how they got there.*

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

A RAG demo answers questions. A RAG *system* has to prove its answers are trustworthy under questioning — which retrieved chunks were right, which answers were actually grounded in them, and where it fails. The evaluation splits into two independently meaningful halves — **Retrieval** (did the right chunks come back in top K?) and **Faithfulness** (given those chunks, is the answer actually grounded in them?) — so a failure always points at the responsible component.

## Eval-Set Construction

20 queries (`eval_set.json`), IDs q001–q020, each with `expected_technique_ids` and/or `expected_cve_ids`, plus a hand-written gold-answer summary. Gold answers are hand-written and verified against ingested chunks — an AI-generated gold would only measure agreement between two models, not correctness.

- **15 scored rows (Group A)** — the reconciliation total used throughout.
- **5 Skip Rows** (q014, q017–q020) — no expected IDs of any kind, never retrieval-scored.
- **q011, q012** — empty `expected_technique_ids` but scored against `expected_cve_ids`; not Skip Rows.

## Metrics

**Precision@3 / Recall@3** — measured against `expected_technique_ids ∪ expected_cve_ids`. Precision's denominator is the actual retrieved count, not a fixed 3 (some paths return fewer). Recall@3 = 0 makes a row Eligibility-Filter-excluded from faithfulness scoring — there's no correct context to be faithful to.

**Faithfulness** — Gemini 2.5 Flash as judge, 1–5 rubric, judge sees only the answer and context (not the gold). Measures grounding, not correctness — an answer can be faithful and still wrong, or refuse and score high for making no claims to contradict (see Known Limitations).

**Faithfulness scores are directional, not precisely comparable across arms.** Repeated observation (q013 flipping 1→5 on identical retrieval and identical refusal behavior — judge variance on an unchanged input) shows single-run judge scores carry enough variance that a 0.5–1.0 point delta between arms should be read as noise unless independently corroborated. Means are reported below for completeness but should not be treated as precise deltas. Full k=3 judge replication on boundary rows (q013, q015) is a documented upgrade path, not yet executed (see Deferred Work).

---

## Current Numbers (as of July 21, 2026 — post-fragmentation-fix, confirmed)

### Retrieval — current architecture (merge + guaranteed-slot rerank + fragmentation resolution)

| Arm | Precision@3 | Recall@3 |
|---|---|---|
| Blind | 0.2444 | 0.5667 |
| Routed, no fixes | 0.2444 | 0.5667 |
| Routed + fixes (merge + guaranteed-slot rerank, pre-fragmentation-fix) | 0.5111 | 0.9444 |
| **Routed + fixes + fragmentation resolution (current)** | **0.5333** | **0.9778** |

Five rows now confirmed at 1.00 recall: q003, q010, q013, q016 (unchanged from before), plus **q015 (0.50 → 1.00)** — the fragmentation fix's one confirmed real fix among the targeted near-ties. q005 unchanged at 0.67 — targeted-recheck confirmed fragmentation is **not** the cause for this row (see Known Limitation #7, revised below). No row regressed.

**Comparison against the pre-rearchitecture (confidence-gated) reference point** — a different pipeline entirely, kept for context, not a like-for-like baseline:

| | Recall | Precision |
|---|---|---|
| Old gated arm (retired architecture) | 0.8222 | 0.4778 |
| Current architecture | **0.9778** | **0.5333** |

Beats the old architecture on both axes simultaneously, same as the pre-fragmentation-fix architecture already did — the fragmentation fix widened that margin further rather than trading one axis for the other.

### Faithfulness — RE-SCORED under the current architecture, July 22, 2026

| Arm | Mean | N Eligible | N Ineligible |
|---|---|---|---|
| Blind | 4.444 | 9 | 6 |
| Routed (pre-fix retrieval, stale) | 5.000 | 9 | 6 |
| Routed + fixes + Confidence Gate (retired architecture) | 4.429 | 14 | 1 |
| **Routed + fixes + fragmentation resolution (current)** | **4.0** | **15** | **0** |

**Full eligibility for the first time — 15/0, not 14/1.** q004, the sole gated-out row under every prior architecture, now retrieves successfully and scores a clean 5. A genuine, checkable structural improvement, not a judge artifact.

**4.0 is not directly comparable to 4.429 as a clean delta.** Three confounds, not one: different N (15 vs. 14), materially different underlying retrieval (context changed twice on July 21 — the rearchitecture, then the fragmentation fix), and — new this pass — two rows that look like judge-calibration errors rather than genuine faithfulness failures (see Known Limitations #5 and #8, below). The pipeline's real faithfulness is plausibly higher than 4.0 once those two are accounted for, but that's a judging-quality question to name, not a number to quietly adjust.

**One row's score was a bug, not a finding, until fixed and re-verified.** q015 initially scored 2/5; the judge's own stated reason included "the answer is cut off mid-sentence" — a `MAX_TOKENS` truncation, not a grounding failure. Root cause: `threat_analyzer.py`'s `GENERATION_CONFIG` capped `max_output_tokens` at 2048, shared between thinking and output tokens with no separate budget — q015's compound cross-corpus answer used 1,966 thinking + 78 output = 2,044/2,048, four tokens from the wall. A second bug compounded it: `_safe_extract_text()` treated `MAX_TOKENS` as an acceptable finish reason, identical to `STOP` — so the truncated answer returned as a clean, unflagged success. Fixed: cap raised to 4096; `MAX_TOKENS` now logs a warning instead of passing silently. q015 alone was re-captured and re-scored (not a full re-run — `generation_capture.py`'s resumability and the judge script's scratch-file resume both made this a single-row operation). Re-verified: 93 output + 1,248 thinking = 1,341 tokens, real headroom; answer reads complete, ends on a full sentence; score moved 2 → 4. The table above reflects the post-fix number.

### Tier Distribution & Cost/Latency (Routed Arm)

13 flash / 7 pro (35% pro), captured under the pre-rearchitecture pipeline. Retrieval-side changes since then don't directly affect tier classification (a routing-layer decision, made before retrieval runs), but this hasn't been re-verified post-rearchitecture or post-fragmentation-fix — still open, unlike faithfulness above.

---

## Architecture (Current Pipeline — Fourth Revision, July 21, 2026)

1. **Route** — `route_query()` picks a corpus (`mitre_only` / `kev_only` / `both` / `skip`) and a model tier (`flash` / `pro`) in one structured-output call. `temperature=0.0` pinned. Unchanged by the retrieval work below.

2. **Retrieve** — `retrieve_for_route()`:
   - **Exact match** — literal CVE/technique IDs in the query fetched directly via ChromaDB metadata filter. Deterministic, unchanged since the original version.
   - **Otherwise:** dense search on the raw query (widened to `config.RERANK_POOL_K=15`) **and** rewrite/decomposition (`retrieve_with_rewrite()`, also widened) **both run unconditionally** — no gate, no threshold deciding which path to trust.
   - **Fragmentation resolution (new this revision)** — before either the guaranteed-slot rerank or the fill-step dedup runs, `_resolve_fragments()` replaces any candidate whose `technique_id` is in the live-computed fragmented set with one candidate built from **all** its chunks, fetched fresh via `db.collection.get()` — not just whichever fragment happened to already be in the pool. `_fragmented_ids(db)` computes the fragmented set once per process from live collection metadata (cached, no persisted file, no drift risk). Only the ~9.8% of technique_ids that are actually fragmented trigger the extra DB call; everything else passes through untouched. Scoring-level fix — never touches `db.semantic_search`, embeddings, or corpus stamps, same isolation contract as the header/crossref strips below. Dedup key (`technique_id`) is unchanged; resolution is a preprocessing step that supplements the existing merge, not a replacement for it.
   - **Guaranteed slot per sub-query** — each sub-query's own best (post-resolution) candidate, reranked against **its own sub-query text** (not the raw query), gets a seat first. Restores a protection `retrieve_with_rewrite()`'s original distance-based merge always had, ported to rerank-score-based — see `docs/lab-notes.md`, July 21 entries, for why this was lost and re-added.
   - **Fill** — remaining slots, from whatever's left (also post-resolution), reranked against the **original raw query** — one shared, comparable scoring surface.
   - Cross-encoder scoring (`reranker.py`, `cross-encoder/ms-marco-MiniLM-L-12-v2`) strips two categories of text before scoring — **never** from what's returned, cited, or generated: (a) structural metadata headers (`TECHNIQUE_ID:`, `TACTIC:`, etc. — embedded verbatim by `ingest.py`), and (b) markdown cross-reference links to other techniques (`[Technique Name](.../techniques/T....)`) — both confirmed causes of the reranker rewarding a chunk for containing the *literal name of the correct answer* inside an unrelated or even negating sentence. See `docs/lab-notes.md` for the specific confirmed cases.
   - **Confidence gate: REMOVED ENTIRELY.** `is_confident()`, `GateNotCalibratedError`, `config.RETRIEVAL_CONFIDENCE_THRESHOLD`, `confidence_gate.py`, `calibrate_confidence.py` are retired. No threshold anywhere in this pipeline.

3. **Generate** — `ThreatAnalyzer.generate_answer()` dispatches to the tier-selected model (`gemini-2.5-flash` / `gemini-2.5-pro`), strict grounding instruction (context-only, exact refusal string on any gap). Unchanged.

4. **Score** — retrieval scored by precision/recall@3 against expected IDs; faithfulness scored by LLM-judge against retrieved context (not gold). All runs logged to MLflow with fail-loud reconciliation invariants and `recall_run_id` lineage tying faithfulness to the exact retrieval run. Arm tag for the fixes arm renamed `routed_with_fixes` → `routed_with_fixes_reranked` so pre- and post-rearchitecture runs aren't conflated under one tag.

---

## Known Limitations

**1. Chunk fragmentation at ingestion — FIXED (scoring-level), July 21, 2026.**

Full-corpus sweep confirmed the scope before the fix shipped:

| | Count |
|---|---|
| Total chunks | 2,140 |
| Unique (corpus, technique_id) entries | 1,823 |
| **Fragmented (>1 chunk)** | **179 (9.8%)** |
| Max fragmentation | T1034 — 7 chunks |

**Fix shipped:** `_resolve_fragments()` concatenates a fragmented technique's chunks (fetched fresh, not just whichever fragment already surfaced in the pool) before dedup/rerank ever sees them — see Architecture, above, for the mechanism. `technique_id` dedup key unchanged.

**Verification:** smoke test (6/6 passing, synthetic data), targeted recheck on the three queries known to be touched by fragmentation, full 15-query suite vs. baseline. Results:

- **q003** — confirmed genuinely fixed, not lucky. Pre-fix, T1140 survived dedup only because its better-scoring chunk happened to be the one selected. Post-fix, T1140 resolves to both chunks merged and wins by a clean 5.3-point margin (9.4927 vs. 4.2064) — a robust win on complete content, not a coin flip.
- **q015** — confirmed fixed, and fragmentation confirmed as the actual root cause (see Limitation #7, revised below).
- **q005** — confirmed **not** fixed by this change; fragmentation ruled out as the cause for this specific row (see Limitation #7, revised below).

**Known residual, not addressed by this fix:** `_fragmented_ids()`'s cache is computed once per process — a long-running process wouldn't notice newly-fragmented entries (e.g. from corpus drift) until restart. Accepted tradeoff for this pipeline's actual usage pattern (re-invoked per eval run, not a long-lived server), not a silent gap. The other ~176 fragmented entries beyond the three targeted this session have not been individually checked for retrieval impact — not investigated, not assumed harmless.

**2, 3, 4. RESOLVED BY REMOVAL, not by fix.** The confidence gate's threshold-precision-loss, unreconciled overlap-zone count, and resubstitution-not-held-out-validation limitations are retired along with the gate itself (July 21, 2026) — there is no threshold left to have these properties. Full rationale in `docs/lab-notes.md`. Not re-numbered below to avoid breaking existing cross-references to #5/#6.

**5. Faithfulness judge is lenient toward wrongful refusal.** An answer with no claims trivially passes the grounding check. Confirmed on q013 and q015 (blind vs. routed: same refusal, scores 1 then 5). Correctness-vs-gold metric remains deferred but justified by data, not yet built.

**New evidence, July 22, 2026, NOT yet reconciled — flagged explicitly rather than merged or dismissed either way.** q009 (current architecture's faithfulness re-score) shows the OPPOSITE direction: a defensible refusal — consistent with the July 18 diagnosis of this exact query as a "legitimate refusal (eval-set defect)," since the query's WBEM framing isn't stated in the retrieved WMI chunk — scored 1, with the judge's own reasoning treating the refusal itself as the unfaithful act. The judge's system instruction explicitly states it is not judging correctness, only grounding; penalizing a no-claims refusal contradicts that instruction on its face. This is the same query-and-mechanism shape as the q003 retraction (see Gated Faithfulness Re-Score, July 15, in `docs/lab-notes.md`) that concluded this limitation was one-directional, not bidirectional — q009 is real evidence the judge can fail in both directions, not just leniency. Deliberately NOT resolved this session: whether q009 is itself noise (same category as q013's documented 1→5 flip) or a stable judge-instruction-following failure is unknown without repetition. Judge k=3 replication (already in Deferred Work) is the natural next step, and directly relevant to the correctness-vs-gold metric already planned — "was this refusal correct" is precisely a correctness question, not a faithfulness one.

**6. Tier classification instability, q003 (gated arm), unresolved.** Unchanged, still open. **Disambiguation, added July 21:** this is unrelated to the *separate* q003 retrieval-near-tie finding (Limitation #7's build history) — same query ID, two different subsystems (routing-layer tier classification vs. retrieval-layer reranking), diagnosed in different sessions. The tier-instability finding concerns whether `route_query()` classifies q003 as `flash` or `pro` across repeated runs; it has nothing to do with which chunks get retrieved for it. Kept separate deliberately, not merged, to avoid conflating two unrelated failure modes that happen to share a query ID.

**7. Reranker cannot reliably discriminate between genuinely similar techniques — REVISED July 21, 2026, post-fragmentation-fix.**

**q015 (T1203 vs. T1190) — resolved, and fragmentation confirmed as the actual cause.** Pre-fix, this was a narrow, non-decisive gap using one incomplete chunk of each (T1203 2.6990 vs. T1190 2.1087). Post-fix, with both techniques scored on their complete, resolved text, T1190 pulls 8.5 points clear (9.5639 vs. 1.1060) — a decisive result. This near-tie really was a fragmentation artifact: T1190 was disadvantaged for lack of its full description, not because T1203 is a legitimately stronger match. This instance of the limitation is closed.

**q005 (T1621 vs. T1111) — NOT resolved by the fragmentation fix; root cause reclassified.** Both T1621 and T1111 were confirmed fragmented and are now scored on complete, resolved text (3 chunks and 2 chunks respectively, fully merged). The gap did not close — it **widened**: 1.39 points pre-fix (8.7969 vs. 7.4099) to 1.65 points post-fix (8.6667 vs. 7.0174). This confirms fragmentation was never the cause here: with both techniques given their full descriptions, the reranker still legitimately prefers T1621 for this sub-query's phrasing. The real cause is the two mechanisms this limitation originally described — (a) genuinely subtle technique-vs-technique distinction that's hard for passage-relevance scoring, and (b) guaranteed-slot budget exhaustion: q005 decomposes into exactly 3 sub-queries, matching `RETRIEVAL_TOP_K=3` exactly, so all 3 guaranteed slots fill before any fill step can run, leaving T1111 with zero path back into the result regardless of its own score. Both remain open and unfixed.

**Also still affects:** the guaranteed-slot mechanism only protects the *single best* candidate per sub-query — a strong runner-up gets no protection at all once it loses that one comparison. When a fill step exists, it falls into a pool still scored against the raw compound query, carrying residual category-suppression bias. When guaranteed slots exactly consume `k` (q005's case), there's no fill step at all.

**Why q005 isn't fixed now:** the same three candidate fixes remain on the table, none newly justified by this session's finding — a different reranker architecture (`bge-reranker-base`, untested), widening the guarantee to top-2-per-subquery when the margin is small, and (now ruled out for this specific row) the fragmentation fix. Consistent with this project's rigor-over-production-ceremony philosophy: documented as a real, understood, and now more precisely diagnosed limitation, rather than patched on a hunch.

**8. Faithfulness judge over-penalizes trivial framing echoed from the query itself (new, July 22, 2026, single instance, not confirmed systematic).**

q006's answer states "The parent technique is called Process Injection" — T1055's chunk fully supports the substantive claim word-for-word ("Adversaries may inject code into processes... execution is masked under a legitimate process"). The word "parent" is not an asserted fact about corpus taxonomy — it's an echo of the query's own phrasing ("What's the parent technique called..."). The judge scored this 1 ("largely fabricated"), which contradicts the rubric's own worked example: trivial connective/framing language that isn't load-bearing is explicitly scored 4, not 1. Distinct failure mode from Limitation #5 — not about refusals at all, about over-strict treatment of query-echoed framing as if it were a new claim. Single instance; whether this is stable miscalibration or one-off variance is unknown without repetition — same k=3 replication named under Limitation #5 would resolve both findings in one pass, not scheduled as two separate efforts.

---

## Deferred Work (not scheduled, not forgotten)

- Fragmentation fix (Limitation #1) — **done, July 21, 2026.** Removed from this list; residual gaps (other 176 entries not individually checked, process-lifetime cache) noted under Limitation #1 above, not treated as open deferred work.
- q005's guaranteed-slot-exhaustion + reranker-discrimination limitation (Limitation #7, q005 half) — still open, now more precisely diagnosed. Candidate fixes: `bge-reranker-base` trial, top-2-per-subquery guarantee when margin is small. Neither pursued without stronger evidence than one query.
- Re-score faithfulness under the current architecture — **done, July 22, 2026.** Mean 4.0, 15/15 eligible (first full eligibility ever recorded). One row (q015) required a generation-side `MAX_TOKENS` fix first — see Faithfulness section above. Two new judge-calibration findings surfaced in the process (Limitations #5's new evidence, #8) — not resolved, carried forward below.
- Correctness-vs-gold metric (separate from faithfulness), to resolve the judge's refusal-blindness (Limitation #5).
- Root-cause q003's gated-arm tier instability (Limitation #6) — note this arm no longer exists post-rearchitecture; whether this finding still applies to the current pipeline is itself an open question.
- Judge k=3 replication on boundary rows — originally q013, q015; **q006 and q009 added, July 22, 2026** (Limitations #5's new evidence and #8) — to quantify variance directly and determine whether either is noise or a stable miscalibration.
- Eval-set growth past 20 queries — repeatedly named as the resolution to several small-n caveats, not yet scheduled.
- Hybrid BM25 — deprioritized further this session: q008/q010's original vocabulary-mismatch framing turned out not to require it (q010's actual failure was header/crossref contamination, both fixed without lexical matching). Still an open question for any query where a genuine coverage gap (not ranking) is eventually confirmed.

## See Also

Full chronological build log, dated findings, retracted claims, and the reasoning behind each fix: `docs/lab-notes.md`.