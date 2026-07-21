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

## Current Numbers (as of July 21, 2026 — post-rearchitecture, confirmed)

### Retrieval — current architecture (merge + guaranteed-slot rerank)

| Arm | Precision@3 | Recall@3 |
|---|---|---|
| Blind | 0.2444 | 0.5667 |
| Routed, no fixes | 0.2444 | 0.5667 |
| **Routed + fixes (merge + guaranteed-slot rerank)** | **0.5111** | **0.9444** |

Computed directly from the 15-row per-row detail of the July 21 full-suite run, not read off a summary line. All four confirmed-fixed rows (q003, q010, q013, q016) score 1.00. q015 scores 0.50, matching Known Limitation #7 exactly — not worse, not better than diagnosed. q005 scores 0.67, up from 0.33 — see note under Limitation #7's discussion of q005 below; not deliberately targeted this session, improvement not fully explained.

**Comparison against the pre-rearchitecture (confidence-gated) reference point** — a different pipeline entirely, kept for context, not a like-for-like baseline:

| | Recall | Precision |
|---|---|---|
| Old gated arm (retired architecture) | 0.8222 | 0.4778 |
| Current architecture | **0.9444** | **0.5111** |

Beats the old architecture on both axes simultaneously — the only run this session to do so; intermediate revisions during the day traded one metric for the other.

### Faithfulness and Tier Distribution — NOT re-scored under the new architecture

The tables below are retained from the pre-rearchitecture pipeline and have not been re-captured against current retrieval output. Faithfulness scoring depends on which chunks actually get retrieved, which changed substantially today — treat these as historical reference only, not current. Re-scoring faithfulness under the new architecture remains in Deferred Work.

## Current Numbers — Faithfulness / Tier (as of July 19, 2026, pre-rearchitecture — see notice above)

### Faithfulness

| Arm | Mean | N Eligible | N Ineligible |
|---|---|---|---|
| Blind | 4.444 | 9 | 6 |
| Routed (pre-fix retrieval, stale) | 5.000 | 9 | 6 |
| **Routed + fixes + Confidence Gate** | **4.429** | **14** | **1** |

Not re-scored under the new architecture — see Deferred Work.

### Tier Distribution & Cost/Latency (Routed Arm)

13 flash / 7 pro (35% pro), captured under the pre-rearchitecture pipeline. Retrieval-side changes since then don't directly affect tier classification (a routing-layer decision, made before retrieval runs), but this hasn't been re-verified post-rearchitecture.

---

## Architecture (Current Pipeline — Third Revision, July 21, 2026)

1. **Route** — `route_query()` picks a corpus (`mitre_only` / `kev_only` / `both` / `skip`) and a model tier (`flash` / `pro`) in one structured-output call. `temperature=0.0` pinned. Unchanged by the retrieval rearchitecture below.

2. **Retrieve** — `retrieve_for_route()`:
   - **Exact match** — literal CVE/technique IDs in the query fetched directly via ChromaDB metadata filter. Deterministic, unchanged since the original version.
   - **Otherwise:** dense search on the raw query (widened to `config.RERANK_POOL_K=15`) **and** rewrite/decomposition (`retrieve_with_rewrite()`, also widened) **both run unconditionally** — no gate, no threshold deciding which path to trust.
   - **Guaranteed slot per sub-query** — each sub-query's own best candidate, reranked against **its own sub-query text** (not the raw query), gets a seat first. Restores a protection `retrieve_with_rewrite()`'s original distance-based merge always had, ported to rerank-score-based — see `docs/lab-notes.md`, July 21 entry, for why this was lost and re-added.
   - **Fill** — remaining slots, from whatever's left, reranked against the **original raw query** — one shared, comparable scoring surface.
   - Cross-encoder scoring (`reranker.py`, `cross-encoder/ms-marco-MiniLM-L-12-v2`) strips two categories of text before scoring — **never** from what's returned, cited, or generated: (a) structural metadata headers (`TECHNIQUE_ID:`, `TACTIC:`, etc. — embedded verbatim by `ingest.py`), and (b) markdown cross-reference links to other techniques (`[Technique Name](.../techniques/T....)`) — both confirmed causes of the reranker rewarding a chunk for containing the *literal name of the correct answer* inside an unrelated or even negating sentence. See `docs/lab-notes.md` for the specific confirmed cases.
   - **Confidence gate: REMOVED ENTIRELY.** `is_confident()`, `GateNotCalibratedError`, `config.RETRIEVAL_CONFIDENCE_THRESHOLD`, `confidence_gate.py`, `calibrate_confidence.py` are retired. No threshold anywhere in this pipeline.

3. **Generate** — `ThreatAnalyzer.generate_answer()` dispatches to the tier-selected model (`gemini-2.5-flash` / `gemini-2.5-pro`), strict grounding instruction (context-only, exact refusal string on any gap). Unchanged.

4. **Score** — retrieval scored by precision/recall@3 against expected IDs; faithfulness scored by LLM-judge against retrieved context (not gold). All runs logged to MLflow with fail-loud reconciliation invariants and `recall_run_id` lineage tying faithfulness to the exact retrieval run. Arm tag for the fixes arm renamed `routed_with_fixes` → `routed_with_fixes_reranked` so pre- and post-rearchitecture runs aren't conflated under one tag.

---

## Known Limitations

**1. Chunk fragmentation at ingestion — CONFIRMED SYSTEMIC via full-corpus sweep, July 21, 2026.**

Original framing (Jul 18): "T1140 only, scope beyond it not established." First revision (earlier July 21, incidental discovery during unrelated diagnostics): "at least 7 techniques." **Both superseded.** A deliberate, full-corpus sweep (one pass over the whole collection, not incidental discovery) found:

| | Count |
|---|---|
| Total chunks | 2,140 |
| Unique (corpus, technique_id) entries | 1,823 |
| **Fragmented (>1 chunk)** | **179 (9.8%)** |
| Max fragmentation | T1034 — 7 chunks |

Roughly **1 in 10** technique/CVE entries in the corpus is split across multiple chunks. This is a structural property of how ingestion chunks source files, not an edge case.

**Directly relevant to Known Limitation #7 (below):** both `T1203` and `T1190` — the two techniques in q015's unresolved near-tie — are fragmented (3 chunks each), confirmed by direct lookup, not inferred.

**Effect on the current architecture:** the merge/dedup step (`_guaranteed_slot_rerank_merge()`) still dedupes by `technique_id` — a fragmented technique's chunks compete against each other for which one survives dedup, using whichever text that specific chunk happens to contain. Confirmed to have resolved favorably by chance at least once (q003: the better-scoring of T1140's two chunks happened to survive) but this is not a guaranteed property of the current design.

**Why not fixed now — explicitly scoped out of this session, not forgotten.** Two candidate fixes were identified but deliberately deferred to a future, separately-scoped session rather than shipped under time pressure at the end of an already-long diagnostic day:

- **Ingestion-level:** re-chunk source files so fragments merge, re-ingest, rebuild the collection. Affects dense search's own embeddings, not just reranking — the widest-blast-radius option.
- **Scoring-level:** concatenate a technique's chunks before cross-encoder scoring, same isolation contract (scoring-only) as the header/crossref strips. Smaller blast radius, but at 9.8% of the corpus this is no longer a "small patch" the way the header fix was — it needs a real design decision about whether to fetch every chunk for any technique appearing in a pool (correct, extra DB round-trips) or only whichever fragments already got retrieved (cheap, still potentially partial).

Both would need the same protocol as every fix shipped today: smoke test, single-query re-check, full-suite re-run, explicit comparison against 0.9444/0.5111 before being kept.

**2, 3, 4. RESOLVED BY REMOVAL, not by fix.** The confidence gate's threshold-precision-loss, unreconciled overlap-zone count, and resubstitution-not-held-out-validation limitations are retired along with the gate itself (July 21, 2026) — there is no threshold left to have these properties. Full rationale in `docs/lab-notes.md`. Not re-numbered below to avoid breaking existing cross-references to #5/#6.

**5. Faithfulness judge is lenient toward wrongful refusal.** Unchanged, still open. An answer with no claims trivially passes the grounding check. Confirmed on q013 and q015 (blind vs. routed: same refusal, scores 1 then 5). Correctness-vs-gold metric remains deferred but justified by data, not yet built.

**6. Tier classification instability, q003 (gated arm), unresolved.** Unchanged, still open. **Disambiguation, added July 21:** this is unrelated to the *separate* q003 retrieval-near-tie finding below (Limitation #7's build history) — same query ID, two different subsystems (routing-layer tier classification vs. retrieval-layer reranking), diagnosed in different sessions. The tier-instability finding concerns whether `route_query()` classifies q003 as `flash` or `pro` across repeated runs; it has nothing to do with which chunks get retrieved for it. Kept separate deliberately, not merged, to avoid conflating two unrelated failure modes that happen to share a query ID.

**7. Reranker cannot reliably discriminate between genuinely similar techniques (new, July 21, 2026).**

Confirmed on q015: `T1203` ("Exploitation for Client Execution") beat `T1190` ("Exploit Public-Facing Application") for the final guaranteed slot by a narrow margin (2.6990 vs. 2.1087 — a real but non-decisive gap, not a landslide). Checked directly: T1203's chunk contains **no** markdown cross-reference to T1190 — this is not the header-contamination or crossref-contamination pattern fixed earlier in the same session. Both techniques are legitimately about "exploiting a software vulnerability to execute code"; the distinguishing detail (public-facing server vs. client application) is real but comparatively subtle for cross-encoder passage-relevance scoring, as opposed to term-matching. **Both `T1203` and `T1190` are confirmed fragmented (3 chunks each — see Limitation #1)** — the near-tie was diagnosed using only one chunk of each, symmetrically incomplete; whether a fragmentation fix would resolve, worsen, or leave this near-tie unchanged is genuinely unknown, not assumed either way.

**A second, distinct manifestation, found while investigating q005 the same day — same root limitation, different mechanism.** q005 decomposed into exactly 3 sub-queries, matching `RETRIEVAL_TOP_K=3` exactly — meaning all 3 guaranteed slots were consumed before any fill step could run. One sub-query guaranteed `T1621` over the expected `T1111` (score 8.7969 vs. 7.4099, a 1.39-point near-tie — same category as T1203/T1190, not a decisive artifact-driven loss). Because no fill step ran, `T1111` had **zero path back into the result** regardless of how it might have scored elsewhere — a sharper version of the "runner-up gets no protection" property below, with no fallback at all when guaranteed slots exactly fill `k`. **`T1111` is separately confirmed fragmented (2 chunks)** — the scored chunk (containing the technique name and opening definition) is plausibly the more competitive half; whether the full, unfragmented text would close a 1.39-point gap is untested.

**Also affects:** the guaranteed-slot mechanism only protects the *single best* candidate per sub-query — a strong runner-up (T1190, T1111) gets no protection at all once it loses that one comparison. When a fill step exists, it falls into a pool still scored against the raw compound query, carrying residual category-suppression bias (visible on q015: an unrelated KEV entry won the third slot over any leftover MITRE candidate). When guaranteed slots exactly consume `k` (q005), there's no fill step at all.

**Why not fixed now:** three candidate fixes now on the table, all explicitly declined pending stronger evidence rather than shipped speculatively: a different reranker architecture (`bge-reranker-base`, untested), widening the guarantee to top-2-per-subquery when the margin is small, and the fragmentation fix itself (Limitation #1) — which both confirmed instances of this limitation now directly implicate as a plausible contributing factor, not yet tested. Consistent with this project's rigor-over-production-ceremony philosophy: documented as a real, understood limitation rather than patched on a hunch.

---

## Deferred Work (not scheduled, not forgotten)

- **NEW, top priority:** fragmentation fix (Limitation #1) — 179/1823 entries (9.8%) confirmed via full-corpus sweep, up from 8 found incidentally. Two candidate approaches identified (ingestion-level re-chunk, or scoring-level concatenation) but deliberately not designed or shipped this session — explicitly deferred to a separately-scoped session rather than built under time pressure. Directly implicated in Limitation #7's two unresolved near-ties (T1203/T1190, T1621/T1111).
- Re-score faithfulness under the current merge+guaranteed-slot-rerank architecture — retrieval was re-run and confirmed (0.9444 recall / 0.5111 precision) July 21; faithfulness was not.
- Correctness-vs-gold metric (separate from faithfulness), to resolve the judge's refusal-blindness (Limitation #5).
- Root-cause q003's gated-arm tier instability (Limitation #6) — note this arm no longer exists post-rearchitecture; whether this finding still applies to the current pipeline is itself an open question.
- Judge k=3 replication on boundary rows (q013, q015) to quantify variance directly.
- Eval-set growth past 20 queries — repeatedly named as the resolution to several small-n caveats, not yet scheduled.
- Hybrid BM25 — deprioritized further this session: q008/q010's original vocabulary-mismatch framing turned out not to require it (q010's actual failure was header/crossref contamination, both fixed without lexical matching). Still an open question for any query where a genuine coverage gap (not ranking) is eventually confirmed.

## See Also

Full chronological build log, dated findings, retracted claims, and the reasoning behind each fix: `docs/lab-notes.md`.