# RAG Evaluation Pipeline — Lab Notes (Chronological Appendix)

*This is the full, dated build history behind `eval_pipeline.md`. It includes findings later revised, retracted, or superseded — left in place rather than deleted, per this project's reconciliation discipline. For current, validated numbers and architecture, see `eval_pipeline.md`. Terminology follows the disambiguation table there: "Skip Rows," "Eligibility Filter," and "Confidence Gate" are used below in place of the earlier overloaded "gate."*

---

## Baseline Numbers (single-pass RAG)

| Metric | Value |
|---|---|
| Precision@3 | 0.2444 |
| Recall@3 | 0.5667 |
| Faithfulness mean | 4.444 (N=9 eligible) |
| Faithfulness-ineligible (recall@3 = 0) | 6 |
| Group A total | 15 (reconciles: 9 + 6) |
| Corpus | chunk_count = 2140 |

Faithfulness mean is inflated by refusals scoring high for making no claims (see judge limitation, below). A baseline to beat, not a headline to celebrate.

## Baseline Failure Analysis

Two classes of failure: **generation** (right context retrieved, answer still failed) and **retrieval** (right chunk never came back).

### Generation failures (Eligibility-Filter-passing rows)

- **q013 — over-refusal (prompt defect).** CVE-2017-0005 present in context; query asks about KEV *provenance*, which no chunk states. Strict grounding forces a whole-query refusal. Judge scored this 1 — refusal contradicts a context that contained the answer.
- **q009 — legitimate refusal (eval-set defect).** T1047/WMI retrieved, but context never says "WBEM." Bridging requires domain knowledge the prompt forbids. Model correctly obeyed instructions; the gold answer assumed a bridge the prompt bans.
- **q015 — mixed.** CVE chunk retrieved but "actively exploited" (KEV provenance) unanswerable from it; T1190 chunk not retrieved at all. Part genuinely unanswerable, part over-refusal.

Through-line: strict no-prior-knowledge grounding trades recall for faithfulness. q013 and q009 are the two poles of that tradeoff.

### Judge limitation — blindness to wrongful refusal

An answer with no claims trivially passes the grounding check. The judge scored q009 and q015 a faithful 5, and only caught q013's wrongful refusal because it happened to reason "contradicts" rather than "no claims." Not a judge bug — an inherent property of grounding-only scoring: faithfulness penalizes hallucination but is blind to under-answering.

### Retrieval failures (Eligibility-Filter-excluded rows, recall@3 = 0)

- **Exact-identifier — q011, q012.** CVE-number-dominated queries; `all-MiniLM-L6-v2` embeds semantic meaning, and an ID carries almost none. Fix candidate: hybrid dense+lexical retrieval (ADR candidate).
- **Multi-hop / compositional — q005, q016.** A single query embedding averages across clauses and matches none strongly enough. Fix candidate: query decomposition.
- **Vocabulary mismatch — q008, q010.** Plain-English queries, technical-term chunks. Fix candidate: query expansion or reranker.

### Summary

| Row | Class | Sub-mode | Fix lands in |
|---|---|---|---|
| q009 | Generation | Legitimate refusal (bad gold) | Eval-set revision |
| q013 | Generation | Over-refusal (strict prompt) | Prompt loosening |
| q015 | Mixed | Over-refusal + retrieval miss | Prompt + routing |
| q005 | Retrieval | Multi-hop | Query decomposition |
| q016 | Retrieval | Multi-hop | Query decomposition |
| q008 | Retrieval | Vocabulary mismatch | Query expansion / reranker |
| q010 | Retrieval | Vocabulary mismatch | Query expansion / reranker |
| q011 | Retrieval | Exact-identifier | Hybrid retrieval (ADR) |
| q012 | Retrieval | Exact-identifier | Hybrid retrieval (ADR) |

---

## Routing Arm Results

`use_routing=True` routes each query to a corpus before retrieval.

| Metric | Blind baseline | Routed | Delta |
|---|---|---|---|
| Precision@3 | 0.2444 | 0.2444 | 0.0000 |
| Recall@3 | 0.5667 | 0.5667 | 0.0000 |
| n_misroutes | — | 0 | — |

Flat delta, zero misroutes: routing selected the correct corpus every time; correct corpus selection didn't move top-3 for 14 of 15 rows (later confirmed directly — see Retrieved-IDs Verification).

### Per-category breakdown

| Corpus | Difficulty | Precision@3 | Recall@3 |
|---|---|---|---|
| cross | hard | 0.1667 | 0.2500 |
| kev | easy | 0.0000 | 0.0000 |
| kev | medium | 0.6667 | 1.0000 |
| mitre | easy | 0.0000 | 0.0000 |
| mitre | hard | 0.3333 | 0.7500 |
| mitre | medium | 0.2667 | 0.8000 |

Both `easy` buckets at 0.0000/0.0000 with 0 misroutes — a retrieval-layer problem, not routing.

### Routed Faithfulness Results

| Arm | Faithfulness Mean | N Eligible | N Ineligible |
|---|---|---|---|
| Blind | 4.444 | 9 | 6 |
| Routed | 5.000 | 9 | 6 |

Decomposes to sums of 40 vs. 45 over the same 9 rows. q013 flipped 1→5 on identical retrieval and refusal — judge variance on ~2 rows, not a system change. **This is the finding that later motivated treating faithfulness deltas as directional only — see `eval_pipeline.md`, Metrics.**

## Routing Stress Test

8 additional queries (`stress_test_routes.py`), routing-only (no recall numbers).

| Query ID | Query | Expected Route | Actual Route | Misroute? |
|---|---|---|---|---|
| q021 | Log4Shell technique + exploited? | both | both | no |
| q022 | Actively exploited + lateral movement | both | both | no |
| q023 | MOVEit + credential access | both | both | no |
| q024 | Unpatched internet-facing software entry | kev_only | mitre_only | **yes** |
| q025 | Tell me about ransomware. | skip\|both | skip | no |
| q026 | T1190 | mitre_only | mitre_only | no |
| q027 | CVE-2021-34473 | kev_only | kev_only | no |
| q028 | What did we just talk about? | skip | skip | no |

0 misroutes on the 6 clean cases. **q024** reproduces the vocabulary-mismatch failure mode at the routing layer — "attackers get in" pattern-matched ATT&CK phrasing despite describing a KEV-style concept. **q025** routed to skip; both skip and both accepted as defensible.

---

## Retrieval Precision-Ceiling Fixes — July 13

**Issue #17.** Three mechanisms added as a new `use_retrieval_fixes` arm:

1. **Exact-match ID lookup** — regex-extracts literal CVE/technique IDs, bypasses the embedder via direct metadata filter. Fully deterministic.
2. **Corpus-tagged query rewrite** — single Gemini call splits a query into 1–4 sub-queries when it chains distinct actions, tags each with a predicted corpus.
3. **Guaranteed-slot cross-corpus merge** — each sub-query's best result gets a guaranteed top-K seat before remaining slots fill globally by distance. Fixes a starvation bug where raw distances aren't comparable across sub-queries.

### Results

| Arm | Precision@3 | Recall@3 |
|---|---|---|
| Blind (pinned baseline) | 0.2444 | 0.5667 |
| Routed (no fixes) | 0.2444 | 0.5667 |
| Routed + fixes | **0.4667** | **0.7556** |

Net: +91% precision, +33% recall vs. baseline from retrieval-layer fixes alone.

### Known Limitation — Residual Dense-Retrieval Ranking Miss (q015, q016)

Both cross-corpus queries still miss one half of their expected answer even after fixes — confirmed present in their sub-query's own candidate pool at low rank (13/15 and 5/15). Genuine embedding-ranking limitation. Pool-widening evaluated, found insufficient on its own. Candidate follow-ups: cross-encoder reranking, chunk-level rewording, hybrid BM25 layer.

### Known Limitation — Rewrite Step Is Not Deterministic Across Runs

Across five runs, the rewrite step (`temperature=0.0`, not guaranteed deterministic per Gemini's floating-point behavior) produced different outcomes run-to-run on q016 (three different outcomes across three runs), q001 (passed then failed after a coincidental keyword split), q004 (regressed after an unrelated prompt-rule addition), q007 (fixed via an explicit "prefer parent technique" rule).

Root cause: the rewrite step is an LLM judgment call sensitive to exact wording variation. The exact-match component (mechanism 1) was correct on every run, no exceptions — instability isolated entirely to the LLM-driven rewrite step.

---

## Fallback-Only Rewrite Restructuring (Confidence Gate) — July 14

**Motivation.** Jul 13's routed+fixes arm improved precision/recall but rewrite ran unconditionally on every non-exact-match query, breaking at least one query (q001) that dense search already answered correctly.

**Design.** `retrieve_for_route()` now tries dense search on the raw query before rewrite, falling through to rewrite only if dense search isn't "confident" — Confidence Gate — top-1 distance at or below a threshold calibrated offline.

**Calibration methodology.**
- Ran dense-search-only over the 15 scored queries; excluded 2 exact-match queries and 0 misroutes — 13 queries used.
- Split top-1 distances by success/failure: 8 successes (0.2328–0.4446), 5 failures (0.324–0.4464) — substantially overlapping.
- Two threshold candidates: coverage-optimized (0.3775) vs. conservative (0.3146, largest success distance below the smallest failure distance).
- **Chose the conservative threshold.** A wrongly-trusted dense result is guaranteed wrong with no rewrite chance to correct it; a correct result sent to rewrite unnecessarily is usually still right. At 0.3775, 3 of 5 observed failures would have been wrongly trusted; at 0.3146, zero are.

**Caveat — resubstitution, not held-out validation.** *(Added July 19, consolidating what was previously stated only as "n=13 is small.")* The threshold was both calibrated on and evaluated against the same 13 queries — the distances used to pick 0.3146 are the same distances later used to judge whether 0.3146 was a good choice. This is a resubstitution estimate, not an out-of-sample one: "zero observed false positives" is true of this exact set by construction, not a validated generalization bound. A single holdout split or leave-one-out cross-validation would produce an honest estimate; deferred given the boundary. See `eval_pipeline.md`, Known Limitations #4, for the consolidated current statement. n=13 being small compounds this (high per-query noise) but is a separate issue from the resubstitution bias itself.

**Precision-loss risk, root-caused Jul 18 — see Known Limitations below.** The threshold is calibrated to full float precision but only ever printed rounded to 4 decimals for manual entry into `config.py`.

**Results — 3 independent runs, frozen pipeline:**

| Run | Path distribution | Precision@3 | Recall@3 |
|---|---|---|---|
| 1 | exact_match: 2, dense_confident: 3, rewrite: 10 | 0.4778 | 0.8222 |
| 2 | identical | identical | identical |
| 3 | identical | identical | identical |

All 15 rows, including all 10 rewrite-path rows, returned byte-identical `retrieved_ids` across all 3 runs.

| Arm | Precision@3 | Recall@3 |
|---|---|---|
| Blind | 0.2444 | 0.5667 |
| Routed, no fixes | 0.2444 | 0.5667 |
| Routed + fixes (ungated, Jul 13) | 0.4667 | 0.7556 |
| **Routed + fixes + Confidence Gate (Jul 14)** ¹ | **0.4778** | **0.8222** |

¹ *Resubstitution estimate — see caveat above and `eval_pipeline.md` Known Limitations #4.*

**q001 — confirmed fixed.** Now resolves via `dense_confident`, never touches rewrite. Deterministic by construction.

**q004 — root cause: not a rewrite problem, a gate-threshold tradeoff.** Widened-pool dense search puts the correct answer at rank 1/15, distance 0.3336 — one of calibration's own recorded "success" distances, sitting in the overlap zone the conservative threshold was built to exclude from trust. Deferred to rewrite, where it's then answered wrong. Not a new bug — the threshold's design explicitly accepted this tradeoff at calibration time. The exact size of the sacrificed-success group ("~4" vs 5 implied elsewhere) has not been reconciled — parked, see Known Limitations.

**q015 — known limitation, not a new regression.** Confident on the `both` route via an unfiltered whole-store search, but only 0.5 recall (misses T1190) — same recall as under rewrite before.

**q016 — the flagship Jul 13 instability example, now stable, and that's itself evidence.** Identical result across all 3 frozen-prompt runs (recall 0.5). The instability previously observed was likely partially an artifact of concurrent prompt editing, not purely sampling — not proven, but the current best explanation. What's exposed once flakiness stops masking it: CVE-2020-0796 sits at rank 5/15 — a dense-ranking miss (Known Limitation, shared root cause with q015).

**mitre|medium delta, resolved.** q001 (documented above) plus q006 (recall already 1.0 under Jul 13's unconditional rewrite; Jul 14's gate lets it bypass rewrite via dense_confident, small precision dip from one extra near-miss chunk, recall unaffected).

---

## Gated Faithfulness Re-Score — July 15

**Motivation.** The Jul 13 "Routed Faithfulness Results" (mean 5.000, N=9/6) were measured against recall@3 = 0.5667. Gating raised recall@3 to 0.8222, so the old N and mean no longer describe the current retrieval arm.

**Precondition, discovered this session.** No code path had ever connected the gated retrieval to answer generation — `run_pipeline` retrieved via plain `semantic_search` regardless of gating. Fixed by adding a `use_confidence_gate` parameter, valid only with `use_routing=True`. A new capture arm (`generation_capture_gated.json`) produced answers actually generated against gated context.

### Results

| Arm | Faithfulness Mean | N Eligible | N Ineligible |
|---|---|---|---|
| Routed (pre-fixes, stale) | 5.000 | 9 | 6 |
| **Routed + fixes + gate (Jul 15)** | **4.429** | **14** | **1** |

Reconciles 14+1=15. Sole ineligible row: q004 (consistent with its Jul 14 diagnosis — second, independent confirmation).

**The mean dropping from 5.000 to 4.429 is not a quality regression.** N grew from 9 to 14 because eligibility tracks recall@3, which jumped 0.5667→0.8222 under gating. A mean over a larger, more honestly-sampled set moving off a suspiciously perfect 5.000 is expected, not evidence of worse answers.

### Judge reliability — lenient toward refusal (revised July 18)

**q013 and q015 both scored 5 again** — the same refusal-trivially-passes pattern, independently confirmed on a different retrieval arm.

**q003 was initially read as an opposite-direction case (over-penalizing a defensible refusal) — that read was wrong.** Root-cause investigation (July 18) found:

- q003 was eligible in *both* the blind and gated arms from the start (recall@3=1.0 both). An earlier draft mis-stated its eligibility history.
- The blind arm's whole-store search retrieved 3 chunks including T1140's worked-example chunk (T1140 is split across two chunks at ingestion — see Known Limitations #1) — full grounded answer, faithfulness 5, legitimately earned.
- The gated arm routed to `mitre_only`, fell through exact-match to the rewrite path (confirmed via `retrieval_path` field), and rewrite returned only T1140's generic-definition chunk plus T1027 — the example chunk never came back. Confirmed stable across 3 independent re-runs.
- Widened-pool inspection shows the example chunk actually ranks #1/15 at distance 0.3146 — nominally tied with the calibrated threshold. `is_confident()`'s comparison (`d <= threshold`) is written to trust a tie; it didn't fire. Most likely explanation: the threshold-truncation issue (Known Limitations #2), not confirmed via `repr()`-level inspection.

**Conclusion: the gated-arm refusal on q003 was correct, not wrongful.** Neither retrieved chunk names a utility — no basis to answer without fabricating. Opposite failure mode from q013/q015, where context contained the literal answer and the model refused anyway.

**q003 is withdrawn as evidence for judge bidirectionality.** An earlier draft used q003 alongside q013/q015 to argue the judge is "inconsistent in both directions." That rested on a misreading of q003's actual retrieved context and is retracted. The judge-limitation finding reverts to its original, one-directional form: lenient toward refusals when context looks topically sufficient to the judge, whether or not it actually was. q003 is reclassified as a retrieval-layer finding (chunk fragmentation + gate-calibration precision loss), not a faithfulness-judge finding.

### Not addressed this run (revised)

Whether q013/q015's refusals should be treated as retrieval-adjacent findings is the same over-refusal tradeoff already characterized in Baseline Failure Analysis, not re-litigated. No change was made to the judge prompt/rubric. q003's root causes are not fixed this pass — logged as known limitation.

---

## Corpus Stamp — Per-Corpus Counts — July 16

**Motivation.** The corpus stamp previously logged one combined chunk count, which can't distinguish a genuine split from any other pair summing to the same total.

**Fix.** `_corpus_stamp()` now reads MITRE and KEV counts separately and reconciles against the collection total before the run proceeds. A mismatch now raises instead of silently producing a combined total.

**Verified.** MITRE and KEV counts sum to the total with no reconciliation failure. Per-corpus counts now logged in MLflow (`corpus_chunk_count_mitre`, `corpus_chunk_count_kev`, `corpus_chunk_count_total`).

## Retrieved-IDs Verification (`compare_retrieval_arms.py`) — July 16

**Motivation.** The Routing Arm Results section's "changed nothing about which three chunks won" claim was inferred from recall parity, never demonstrated chunk-by-chunk.

**What was built.** A standalone script comparing `retrieved_ids` row-by-row between two arms, classifying each row `IDENTICAL`, `SAME_SET_DIFF_ORDER`, or `DIFFERENT`.

**Result:**

| Status | Count |
|---|---|
| Identical | 14 |
| Same set, different order | 0 |
| Different sets | 1 |

**q013 is the sole exception** — both arms retrieve the two correct CVEs in the first two slots; the third slot differs (blind pulls a structurally-irrelevant MITRE technique, routed swaps in a different wrong CVE). Both score equally as "not expected" so neither metric moved, but the retrieved chunk itself is not the same chunk.

**Revised claim.** "Routing changed nothing about top-3" is corrected to "changed nothing *material*, verified true for 14 of 15 scored rows, not universally true." Not investigated further: whether q013's specific swap reflects a systematic property of `kev_only` filtering or is a one-off.

---

## Tier Classification Added to Router — July 17

**Motivation.** Routing decided which corpus but had no signal for how much reasoning a query needs.

**What was built.** `ModelTier` (`flash`/`pro`) added to `RoutingDecision` alongside `route`, one structured-output call answers both. `route=skip` forces `tier=flash` in code, not just by prompt example.

**Sanity-check tooling.** `inspect_routes.py` gained a display-only `tier` column (no ground truth to score against).

**Finding: the routing call had no `temperature` pinned.** First full pass classified 5/20 as `pro`; a targeted re-run of q002 alone returned `flash` with no edits in between. Root cause: `route_query()`'s `generate_content` call never set `temperature`. This gap predates tier — route's own classification sits far enough from any decision boundary that the gap never surfaced; tier's boundary is narrower. Fixed via `temperature=0.0`. q002 confirmed stable at `pro` across 3 repeated runs post-fix.

**Caveat inherited, not new.** `temperature=0.0` narrows variance but doesn't guarantee determinism (floating-point ops across GPU cores) — same property surfacing on a second call site.

**Post-fix boundary case found July 18: q003.** Observed classifying `flash` on original capture and `pro` on a same-day repeat, no code/prompt change in between — same instability category q002 showed pre-fix, on a query the fix wasn't specifically tested against. Not yet root-caused; only two data points exist. Open item.

## Tier Dispatch Wired to Generation — July 17

**What was built.** `generate_answer()` gained a `tier` parameter; model selection moved from once-per-instance to once-per-call via a `_TIER_MODEL` map. `PRO_MODEL_NAME` defaults to `gemini-2.5-pro`. Blind arm passes `tier=None` → resolves to `FLASH`, byte-identical to pre-tiering behavior.

**Verified, not assumed.** A standalone script confirmed routing's computed tier and generation's actually-dispatched model agree end-to-end for both a known-pro (q005) and known-flash (q001) query.

## Tier-Tagged Capture — Distribution and Stability — July 17

**Interface change.** `run_pipeline()` return signature changed from 3-tuple to 4-tuple, surfacing tier past the function boundary.

**Schema-safety fix.** `generation_capture.py`'s `_load_existing()` previously reused any successful row regardless of fields — meaning pre-tier captures would've been silently reused as a mixed artifact. Fixed via a `REQUIRED_ROW_KEYS` check; confirmed working (stale artifact triggered full regeneration of all 20 rows).

**Fresh capture — tier distribution.**

| Source | Pro count | Pro rate |
|---|---|---|
| Jul 17 spot-check (pre-fix) | 5/20 | 25% |
| Jul 17 full capture (post-fix) | 7/20 | 35% |

**Not a before/after drift measurement — the first is superseded, not compared against.** The spot-check ran before `temperature` was pinned; reading 25%→35% as drift over time would repeat the same cross-run comparison error the reconciliation discipline exists to prevent (same shape as the Jul 15 5.000-vs-4.429 mistake). Correct read: two queries (q003, q013) resolve differently under the pinned call, both verified stable across 3 repeated runs each. Current known distribution: 13 flash / 7 pro (35% pro).

**(Note added July 18: a separate repeat of q003's gated-arm capture on a different day showed flash → pro → pro across 3 runs — see the Jul 18 boundary-case note above. Whether this contradicts the 3-run stability claimed here has not been reconciled; flagged open.)**

**Capture results.** 20/20 rows regenerated (0 reused — all schema-stale), 0 failed, 4 Skip Rows (q017–q020, all `tier=flash` as enforced). Route distribution unchanged (0 misroutes).

### Cost/Latency-Per-Tier Table

**Design choice: thinking left on** for the answer-generation call, unlike every other call site (router, rewrite, judge), which explicitly disable it. Numbers below reflect real production behavior, not a controlled comparison — meaning latency/cost could vary run-to-run for the same non-determinism reason tier classification did.

**Pricing** (pulled 2026-07-17, Developer API Standard tier): Flash $0.30/$2.50 per 1M input/output tokens; Pro $1.25/$10.00. Not confirmed identical to Vertex AI billing — directionally correct, not final.

**Results** (16 usable rows, excludes 4 Skip Rows):

| Tier | N | Avg latency | Avg cost | Total cost | Avg input tok | Avg output tok | Avg thinking tok |
|---|---|---|---|---|---|---|---|
| flash | 9 | 2.74s | $0.00105 | $0.00941 | 647 | 25 | 316 |
| pro | 7 | 11.37s | $0.01243 | $0.08703 | 585 | 80 | 1091 |

**Hypothetical — same 16 calls, uniform tier** (holds token counts fixed, re-prices; an approximation):

| Scenario | Total cost |
|---|---|
| Actual (tiered) | $0.09643 |
| All-Flash | $0.03111 |
| All-Pro | $0.12494 |

Tiering saves 22.8% vs. all-Pro, costs 210% more than all-Flash — modest savings, since 44% of usable queries still land on the ~12x-more-expensive tier.

**Where the 12x comes from.** Pro's per-token price is 4x Flash's on both input/output, and Pro produced 3.5x more thinking tokens on average — compounding, not one factor.

**Latency.** Pro averages 4.15x Flash's latency — a real user-facing cost, not just billing.

**What this table does not show.** Whether Pro's answers are actually better than Flash's on those 7 queries — no faithfulness/correctness comparison exists between tier-generated answers on the same queries yet.

**Not yet run: tiered faithfulness re-score.** The Jul 15 re-score predates tier dispatch (Jul 17) entirely.

---

## Known Limitations — July 18 (original entries; see `eval_pipeline.md` for the current consolidated list, further revised July 21)

### 1. Chunk fragmentation at ingestion (T1140, scope beyond it unknown)

T1140.txt is split across two separate chunks — confirmed by direct query, not inferred. One carries the generic definition, the other the worked example. The split falls at a paragraph boundary, consistent with a chunking strategy that doesn't recognize an example as content that should stay attached to its preceding sentence.

**Effect.** A query needing the example specifically (q003) fails whenever only the definition chunk is retrieved, but recall@3 still reports success — recall is scored at the `technique_id` label level and can't see a technique's content is split across chunks.

**Scope not established.** T1140 is the only technique confirmed so far, found while investigating one query's failure, not a systematic sweep. *(Superseded July 21 — see below and `eval_pipeline.md`, Known Limitation #1: a full-corpus sweep found 179/1823 entries, 9.8%, fragmented; the fix itself shipped later the same day — see "Fragmentation Fix — Design, Implementation, and Verification," below.)*

**Why not fixed now.** Requires an ingestion-time change or a recall-metric change; both larger than a same-day fix, out of scope for the boundary.

### 2. Confidence-gate threshold precision loss

`calibrate_confidence.py` computes a full-precision threshold but only ever prints it rounded to 4 decimals, with a human hand-copying the value into `config.py`. No code path writes the exact value programmatically.

**Observed effect.** On q003, the best candidate sits at distance 0.3146 to 4-decimal precision — apparently tied with the stored threshold. `is_confident()`'s comparison is written correctly and would trust a genuine tie; in production this row still fell through to rewrite, consistent with the stored threshold being a rounded-down truncation. Not confirmed via direct `repr()`-level inspection — best available explanation, not proven.

**Why it matters beyond q003.** Systematic risk: any query whose top-1 distance lands within rounding distance of the true threshold is subject to the same silent misclassification, in either direction.

**Why not fixed now.** Low-risk, mechanical fix; not because it's difficult. *(Moot as of July 21 — the confidence gate and its threshold are retired. See below.)*

### 3. Overlap-zone success count, unreconciled (parked)

The Jul 14 q004 write-up refers to "~4" sacrificed successes; direct arithmetic from the same section's own numbers implies 5. Not reconciled — could be an approximation in original wording, or a genuine calibration-time vs. production-time search difference (the `both`-route caveat already named in `confidence_gate.py`'s docstring). Parked deliberately; the original success/failure distance lists would resolve this if revisited. *(Moot as of July 21 — no overlap zone concept exists without the gate. See below.)*

---

## Confidence Gate Retirement + Merge/Rerank Redesign + Diagnostic Session — July 21, 2026

### Context

`eval_pipeline.md`'s Known Limitations #2 (threshold precision loss), #3 (unreconciled overlap-zone count), and #4 (resubstitution, not held-out validation) all traced back to the confidence gate's calibrated distance threshold. Rather than fix each individually, the gate was retired entirely and replaced with a cross-encoder reranker — removing the threshold removes all three limitations as a structural byproduct, not three separate patches.

### Architecture, three revisions in one session

**Revision 1 (flat merge+rerank).** Exact-match unchanged. For every other query: dense search (widened to `RERANK_POOL_K=15`) and rewrite/decomposition (also widened) both run unconditionally — no gate deciding which to trust. Both pools merged, deduped by `technique_id`, reranked ONCE against the raw query (`cross-encoder/ms-marco-MiniLM-L-6-v2`). MLflow arm tag renamed `routed_with_fixes` → `routed_with_fixes_reranked` so pre/post-rearchitecture runs aren't conflated.

Diagnostic tooling built alongside: `inspect_candidate_pool.py` — pulls raw dense and rewrite pools *before* dedup/merge (something the production pipeline's own `candidate_pool` field, captured post-dedup, cannot show), checks whether an `expected_id` is present anywhere in the raw union (coverage) vs. present but not in top-k (ranking), and prints full-width reranked scores (not just top-3) so a losing candidate's actual rank and score gap is visible.

**Three real diagnoses from Revision 1, run against q003, q010, q015:**

| Query | Failure | Mechanism |
|---|---|---|
| q003 | T1140 (expected) lost a near-tie at rank #5, score -1.1309, spread of 0.02 among 3 candidates just below the top-3 cutoff | Initially hypothesized as dedup collision (see Retraction, below) — actual cause: scored against the raw query instead of canonical phrasing |
| q010 | T1105 (expected) lost decisively (rank #5, -2.4754, 4.91 behind the leader) to T1659/T1650 | Root-caused: `ingest.py` embeds header lines (`TACTIC: Initial Access`) verbatim as scoreable chunk text; query's own "Once initial access is achieved..." phrasing matched the header literally |
| q015 | T1190 (expected) lost decisively (rank #18/25, -3.7646, 8.86 behind the leader) — **every** MITRE technique in the pool scored negative, every CVE but two scored positive | Category-level suppression: compound query (CVE half + MITRE half) scored as one string; a 100%-MITRE chunk scores poorly against a CVE-dominated query regardless of which specific technique it is |

**Retraction, mirroring the project's existing q003-eligibility precedent (Jul 15 entry, above):** q003's failure was initially attributed to `_dedupe_merge()` colliding T1140's two known-fragmented chunks (definition vs. worked-example — Known Limitation #1), on the reasoning that both chunks could plausibly enter the merged pool from different sources and only one survive dedup. **This was tested directly via `inspect_candidate_pool.py` and disconfirmed:** T1140 was present and intact in the post-dedup pool (position 1 of 19); dedup was not the cause. The actual cause (below) is unrelated to fragmentation. Flagged here explicitly, same as q003's earlier CVE-eligibility retraction, rather than silently corrected.

**Incidental finding, independent of any single query's diagnosis:** chunk fragmentation (same `technique_id`, multiple distinct chunks) confirmed on 5 additional techniques beyond T1140 during these diagnostic runs — T1027, T1202 (found via q003), T1650, T1659 (found via q010), T1105, T1570 (found via q010's later re-run). Not a targeted sweep; found by accident while diagnosing unrelated failures. Updates Known Limitation #1's scope. *(Superseded later the same day — see Full-Corpus Fragmentation Sweep, below.)*

### Fix 1 — Guaranteed-slot restoration (Revision 2)

**Root realization:** `retrieve_with_rewrite()`'s own original merge logic already had a guaranteed-slot mechanism (each sub-query's best-by-distance result seated first, preventing one sub-query's results from starving another's). Revision 1's flat merge silently discarded this — treating rewrite's output as "just another pool to merge and rerank" lost the protection entirely. This is plausibly why q010 broke under the new architecture in the first place (the old gated arm got q010 right via this exact mechanism, using distance).

**Implementation.** `query_rewrite.py`'s `retrieve_with_rewrite()` gained `return_subquery_pools` (additive, default `False`, existing callers byte-identical) — exposes each sub-query's raw candidates *before* that function's internal merge collapses them. `retrieval_pipeline.py`'s new `_guaranteed_slot_rerank_merge()`: each sub-query's own best candidate, reranked against **its own sub-query text**, gets a seat first (cross-sub-query score comparability explicitly not assumed, same caution the original distance-based version documents). Remaining slots filled from whatever's left, reranked against the **original raw query** — one shared, comparable surface.

**Result: q003 fixed, confirmed.** T1140 scored 6.4317 against its rewritten sub-query text (a MITRE-canonical rephrasing per `rewrite_query()`'s own Rule 3) vs. 5.5268 for the runner-up — a clean win, not a near-tie. Confirms the hypothesis that canonical rephrasing, not just wider pools, was what q003 needed.

**Result: q015 partially improved.** T1190 moved from -3.7646 (Revision 1, scored against the compound raw query) to +2.0278/-1.0634 across two sub-query-scored runs — no longer category-suppressed. Did not fully resolve (see Fix 3 and Residual, below).

### Fix 2 — Metadata header stripping

**Root cause, confirmed on q010.** `ingest.py` embeds raw `.txt` file text verbatim into chunks — `TECHNIQUE_ID:`, `TACTIC:`, `DATE_ADDED:`, `Technique Name:`, `Platforms:` (MITRE format) and `VULNERABILITY_ID:`, `Vulnerability Name:`, `Vendor:`, `Product:`, `Patch Due Date:` (KEV format) are literal, scoreable chunk content, not stripped metadata. Query "Once initial access is achieved..." matched T1659's chunk on the literal line `TACTIC: Initial Access` — pure header-phrase overlap, unrelated to the chunk's actual content about content injection.

**Implementation.** `reranker.py`: `_strip_metadata_header()`, a regex over the observed field-label set (not a repo-wide sweep of `threat_reports/*.txt` — widen if a future run surfaces an uncovered field), applied only to the text passed into the cross-encoder's scoring pairs inside `rerank()`. The text returned in `rerank()`'s output tuples — and therefore everything citations/generation ever see — is untouched. Confirmed via smoke test (`rerank()` output equals original input, not the stripped version).

**First re-test of q010 (with header stripping, still on L-6-v2): still failed.** T1570 beat T1105, but for a *different* reason this time — worth naming this as a genuine second mechanism, not a failed fix. Led to the model-capacity test, below.

### Model swap test — ms-marco-MiniLM-L-6-v2 → L-12-v2

**Motivation.** After header-stripping, T1570 (wrong) still beat T1105 (correct) on q010 — not a near-tie (5.8905 vs. -0.9483, a real gap). Hypothesis: capacity ceiling in the smaller model.

**Test design.** Same MS MARCO family/training data, more capacity — deliberately not jumping to a different architecture (`bge-reranker-base`) yet, to isolate capacity as a single variable.

**Result: hypothesis REJECTED, and the data pointed the opposite direction.** T1570's score rose from 5.8905 → 7.0617 (+1.17); T1105's rose only -0.9483 → -0.4962 (+0.45). The larger model was *more* confident in the wrong answer, not less — ruling out "insufficient capacity" and suggesting a real, strong textual signal was driving the score (which a bigger model would naturally weight more, not less).

### Fix 3 — Technique cross-reference stripping

**Root cause, confirmed on TWO independent technique pairs, checked directly against real chunk text (not inferred).**

- **T1570 → T1105:** T1570's chunk contains `"(i.e., [Ingress Tool Transfer](https://attack.mitre.org/techniques/T1105))"` — T1105's exact canonical name and ID, embedded as a loose parenthetical association.
- **T1189 → T1190:** T1189's chunk contains `"Unlike [Exploit Public-Facing Application](https://attack.mitre.org/techniques/T1190), the focus of this technique is..."` — T1190's exact canonical name and ID, embedded inside an **explicit negation**. T1189 still won the guaranteed slot over T1190 despite this, ruling out any hope that "the model probably ignores links in dismissive/contrastive framing" — it doesn't reliably use surrounding grammar as a signal at all.

MITRE ATT&CK write-ups routinely cross-reference related techniques this way; this is a structural property of the source data, not a one-off collision.

**Implementation.** `reranker.py`: `_strip_technique_crossrefs()` — strips the entire markdown link construct (bracketed name AND url) for `/techniques/` links (parent and sub-technique paths, e.g. `T1027` and `T1027/010`), applied uniformly regardless of surrounding phrasing (per the negation finding, above). Scope confirmed only for `/techniques/` links — `/software/`, `/groups/`, `/tactics/` links deliberately left untouched, not confirmed as a problem. Verified via smoke test against the actual T1570 and T1189 chunk text (not synthetic stand-ins): both cross-references correctly removed, real surrounding content preserved, `rerank()`'s returned text still the original unstripped version.

**Result: q010 fixed, confirmed with real data** T1105 -1.8027 vs. T1570 -3.5625 — a clean 1.76-point margin.

**Result: q015's T1189 threat resolved, confirmed.** T1189 dropped from beating T1190 by 6.5 points to losing by ~1.4–3.4 points across runs. **But a third, different competitor (T1203) then won the guaranteed slot**, by a narrow 0.5909-point margin (2.6990 vs. 2.1087) — not a landslide, and not decisively resolved by this fix.

### q015 residual — T1203 vs. T1190, documented as Known Limitation #7

**Checked directly:** T1203's chunk contains no markdown cross-reference to T1190. Not the header or crossref pattern. T1203 ("Exploitation for Client Execution") and T1190 ("Exploit Public-Facing Application") are legitimately similar techniques — both fundamentally "exploit a vulnerability to execute code" — with a real but comparatively subtle distinguishing detail (client vs. public-facing server) for passage-relevance scoring specifically.

**Decision: document, don't further engineer against, absent stronger evidence.** Two candidate fixes considered and explicitly declined: (1) `bge-reranker-base` — a genuinely different architecture/training-data test, not ruled out, just not pursued without more justification than one query; (2) widening the guarantee to top-2-per-subquery when the score margin is small — speculative, no confirmed mechanism to justify building it. Consistent with this project's rigor-over-production-ceremony philosophy. Full writeup: `eval_pipeline.md`, Known Limitation #7. *(Revisited and partially resolved later the same day — see Fragmentation Fix, below.)*

### Fix 2, refined — administrative vs. descriptive-title header fields

**Regression found via full-suite run, not targeted diagnosis.** After Fix 3 landed, a full-suite eval run (below) surfaced q013 regressing to 0.00 recall — never individually diagnosed until the full run caught it, three architecture revisions after it started silently degrading (1.00 → 0.50 → 0.00, invisible because no targeted diagnostic run happened to include q013).

**Root cause, confirmed on real chunk text.** Fix 2's header-stripping treated `Technique Name:`/`Vulnerability Name:` the same as purely administrative fields (`TACTIC:`, `DATE_ADDED:`, etc.) — full-line removal. But these two fields carry the technique/vulnerability's actual descriptive title, not category metadata. `CVE-2017-0005`'s `Vulnerability Name` is *"...GDI Privilege Escalation Vulnerability"* — with that line gone, its body only says *"gain privileges"* (paraphrase, not the query's literal phrase), while a wrong candidate (`CVE-2014-1812`) happened to restate "privilege escalation" verbatim in its own body and won purely on that coincidence.

**Implementation.** Split `_strip_metadata_header()` into two passes: full-line removal for `TECHNIQUE_ID`, `VULNERABILITY_ID`, `TACTIC`, `DATE_ADDED`, `Platforms`, `Vendor`, `Product`, `Patch Due Date` (unchanged, still catches q010's original `TACTIC:` confound); label-only removal for `Technique Name`/`Vulnerability Name` — the value now survives scoring. Verified via smoke test that both the q010 confound (`TACTIC: Initial Access`, fully stripped) and the q013 regression (`Vulnerability Name` value, now preserved) are handled correctly by the same function.

**Result: q013 fixed, decisively.** `CVE-2017-0005` 9.1439 vs. runner-up 8.9170 (`CVE-2017-0001`, the other expected ID) vs. third-place 6.4243 — both expected IDs took the top two slots with a real gap to anything else.

**Result: q010 re-confirmed, no regression from the refinement.** T1105 9.2137 vs. T1570 0.9764 — an 8.2-point margin, wider than the 1.76-point margin at the original Fix 3 confirmation. Checked explicitly rather than assumed safe, since q010's original confound (`TACTIC:`) is untouched by this change but "should be unaffected" had already proven an unsafe assumption once this session (the model-swap caching scare).

### Full-suite run — first real confirmed numbers post-rearchitecture

With all three fixes in place (guaranteed-slot restoration, crossref stripping, refined header stripping), a full 15-row eval run:

| | Recall | Precision |
|---|---|---|
| Blind / routed (unaffected by any of this) | 0.5667 | 0.2444 |
| This session, flat merge (before today's fixes) | 0.7222 | 0.4000 |
| Old gated arm (retired architecture, different pipeline) | 0.8222 | 0.4778 |
| **Current architecture, final** | **0.9444** | **0.5111** |

First run all session to beat the old pre-rearchitecture reference point on both axes simultaneously — earlier intermediate states had traded one metric for the other. q003, q010, q013, q016 all confirmed at 1.00 recall. q015 at 0.50, matching Known Limitation #7 exactly.

**q005 improved (0.33 → 0.67) without being targeted by any fix this session** — noted as an open, unconfirmed hypothesis (plausibly the same header-refinement mechanism helping T1078 compete on its own canonical name), not claimed as understood.

### q005 investigation — a second manifestation of Limitation #7

**Motivation.** The unplanned q005 improvement above prompted a direct check, same protocol as every other query this session, rather than accepting an unexplained gain without diagnosis.

**Finding: guaranteed-slot budget exhaustion, not a text artifact.** q005 decomposed into exactly 3 sub-queries — matching `RETRIEVAL_TOP_K=3` exactly. All 3 guaranteed slots filled (sub-queries 2 and 3 correctly guaranteed `T1078` and `T1098`; sub-query 1 guaranteed `T1621` over the expected `T1111`, an 8.7969 vs. 7.4099 near-tie — comparable margin to T1203/T1190, not a decisive artifact loss). Because all `k=3` slots were already guaranteed, **the fill step never ran at all** — `T1111` had zero path back into the result regardless of how it might have scored. A sharper version of Limitation #7's "runner-up gets no protection" property: when guaranteed slots exactly consume `k`, there isn't even a fill step to fall back on.

**Text check, same rigor as every other diagnosis this session:** pulled T1111's actual chunk directly — no markdown cross-reference to T1621 or any other technique. Rewrite's sub-query text ("Bypass Multi-Factor Authentication") is neutral, containing neither T1621's nor T1111's canonical name — ruled out as a cause.

**Discovered via this check: T1111 is fragmented (2 chunks)**, confirmed via a new tool built specifically to check this and a second hypothesis (cross-encoder token truncation) directly rather than inferring from possibly-stale pasted text. Truncation ruled out cleanly — both T1111 chunks (181 and 218 tokens combined with the query) are well under the model's 512-token limit. Fragmentation, not truncation, is the more likely contributing factor — though whether fixing it would close a 1.39-point gap remains untested. *(Tested later the same day — see Fragmentation Fix, below: it does not close the gap, it widens it slightly.)*

### Full-corpus fragmentation sweep

**Motivation.** T1111's fragmentation, found by accident investigating q005, prompted the question already implicit after 7 incidental discoveries: how big is this actually? Built a standalone script — one `db.collection.get()` call over the whole collection, grouped by `(corpus, technique_id)`, counting chunks per entry. Read-only, zero risk to any metric.

**Result: far larger than incidental discovery suggested.**

| | Count |
|---|---|
| Total chunks | 2,140 |
| Unique (corpus, technique_id) entries | 1,823 |
| Fragmented (>1 chunk) | **179 (9.8%)** |
| Max fragmentation | T1034 — 7 chunks |

**Directly checked, per the open question from Limitation #7's writeup:** `T1203` and `T1190` (q015's unresolved near-tie) are both fragmented, 3 chunks each. Whether a fragmentation fix helps, worsens, or doesn't move this specific near-tie is unknown — both sides of the comparison are equally affected, not confirmed to favor one direction.

---

## Fragmentation Fix — Design, Implementation, and Verification — July 21, 2026 (continued)

### Context

Full-corpus sweep re-confirmed the same day: 179/1823 (9.8%) fragmented, no drift since the earlier count. `T1621` (q005's competing candidate) checked specifically at this point and confirmed fragmented (3 chunks) — not previously known, materially changes the q005 diagnosis: the earlier "T1621 beat T1111" comparison was never a fair fight between complete descriptions, on *either* side.

### Design decisions

Two open questions from the sweep-writeup handoff, both decided before any code was written, per this project's no-guessed-interfaces discipline:

1. **Scoring-level fix, not ingestion-level.** Ingestion-level (re-chunk source files, rebuild the collection) would shift dense search's own embeddings, not just reranking — full downstream re-verification of every fix this pipeline has shipped. Scoring-level is containable entirely inside `_guaranteed_slot_rerank_merge()`, same isolation contract as the header/crossref strips: changes what the scorer sees, never what's returned/cited/generated.
2. **Targeted fetch, not fetch-everything, not pool-only.** Three options considered: (a) fetch every chunk for any technique_id appearing in a pool, always — correct, but pays a DB round-trip even for the 90.2% of technique_ids that are never fragmented, on every query, at generation time, not just in eval batches; (b) concatenate only whichever fragments already happened to surface in the pool — free, but not actually a fix, same chance-based completeness q003 already had; (c) **chosen:** compute the fragmented set live from collection metadata once per process (`_fragmented_ids()`, cached, no persisted file — avoids the drift risk a hand-maintained list would reintroduce), and only fire the extra fetch for technique_ids actually in that set. Full completeness where needed, zero added cost everywhere else.
3. **Dedup key unchanged.** `technique_id` stays the dedup key. Fragment resolution (`_resolve_fragments()`) runs as a preprocessing step before both the guaranteed-slot rerank and the fill-step dedup — supplements the existing merge logic rather than replacing it.

### Implementation

`_fragmented_ids(db)` and `_resolve_fragments(db, candidates)` added to `retrieval_pipeline.py`. `_guaranteed_slot_rerank_merge()` signature gained a leading `db` parameter; its first two lines now resolve `dense_candidates` and every sub-query pool's candidates before any dedup/rerank logic runs. `retrieve_for_route()`'s call site updated to pass `db` through. Module docstring's stale fragmentation-scope note ("at least 5 techniques") corrected to the confirmed 179/1823.

**Known accepted tradeoff, not a gap:** `_FRAGMENTED_IDS` is a module-level cache, computed once per process. A long-running process wouldn't notice newly-fragmented entries until restart — acceptable given this pipeline is re-invoked per eval run, not a long-lived server.

### Smoke test

`tests/smoke_test.py` — 6 tests, synthetic data, `rerank()` stubbed to a deterministic length-based score so assertions don't depend on real relevance. One test bug found and fixed before all 6 passed: `test_resolve_fragments_dedupes_repeated_technique_in_pool` initially asserted exactly 1 DB call for a repeated-technique-in-pool case, but didn't pre-warm `_fragmented_ids()`'s own cache first — the assertion was actually counting 2 calls (one to populate the cache, one to fetch the fragment set), not a code bug. Fixed by pre-warming the cache before capturing the call count, isolating the actual invariant under test. All 6 passing after the fix.

### Diagnostic tooling update

`inspect_candidate_pool.py` (recovered from local storage, not rebuilt from scratch) needed two changes before it was safe to run: (1) its call to `_guaranteed_slot_rerank_merge()` needed the new `db` parameter — would otherwise raise `TypeError` cleanly, not silently misbehave; (2) more materially, its "guaranteed slot preview" section reimplements the guarantee mechanism for visibility by reranking each sub-query's raw candidates directly — without also calling `_resolve_fragments()` first, this preview would show the pre-fix near-tie even once the fix was live, silently disagreeing with what production actually returns. Fixed to resolve fragments before the preview rerank, matching production exactly. Also added: known-fragmented-id flags on each raw pool listing, and `(resolved, N chunks merged)` annotations on any candidate that went through resolution — needed to actually see the fix acting during diagnosis, not just infer it from score deltas.

### Targeted recheck

Three queries confirmed touched by fragmentation: q003, q015, q005.

**q003 — confirmed genuinely fixed, not lucky.** T1140 now resolves to both its chunks merged, scores 9.4927, and beats T1027 (also resolved, 3 chunks merged) by a 5.3-point margin (4.2064). Pre-fix, T1140's survival depended on which single chunk happened to win dedup; post-fix, it wins decisively on complete content.

**q015 — confirmed fixed, fragmentation confirmed as the actual root cause.**

| | Pre-fix (partial chunk) | Post-fix (full text) |
|---|---|---|
| T1190 | 2.1087 | 9.5639 |
| T1203 | 2.6990 | 1.1060 |

Pre-fix, T1203 was narrowly ahead. Post-fix, T1190 pulls 8.5 points clear. This resolves Known Limitation #7's q015 half: the near-tie really was a fragmentation artifact, not a genuine reranker-discrimination limit.

**q005 — confirmed NOT fixed; root cause reclassified, not papered over.**

| | Pre-fix (partial chunk) | Post-fix (full text) |
|---|---|---|
| T1621 | 8.7969 | 8.6667 |
| T1111 | 7.4099 | 7.0174 |
| Gap | 1.39 | 1.65 |

Both techniques now score on complete, resolved text — the mechanism worked exactly as designed. The gap **widened**, not closed, confirming the risk flagged before this fix was written: T1621 being separately fragmented meant a naive fix could strengthen it as much as T1111. It did. This rules fragmentation out as q005's cause: with both descriptions complete, the reranker still legitimately prefers T1621 for this sub-query phrasing. The real cause remains Known Limitation #7's other two mechanisms — genuine reranker-discrimination difficulty between similar techniques, and guaranteed-slot budget exhaustion (q005's 3 sub-queries exactly fill `k=3`, so no fill step ever runs, leaving T1111 with zero path back regardless of score). Both still open, unaddressed by this fix, and not expected to be addressed by it — this was checked, not assumed.

### Full-suite verification

| | Baseline (pre-fragmentation-fix) | Post-fix | Δ |
|---|---|---|---|
| Recall | 0.9444 | **0.9778** | +0.0334 |
| Precision | 0.5111 | **0.5333** | +0.0222 |

Improves both axes simultaneously, same bar the rearchitecture itself cleared. Row-by-row: q015 0.50 → 1.00 (largest single contributor to the recall gain, consistent with the targeted recheck); q005 0.67 → 0.67, unchanged, consistent with fragmentation being ruled out as its cause; q003, q010, q013, q016 unchanged at 1.00, no regressions anywhere. Reconciliation check passed (recall sums match; scored + gated == total).

**Go/no-go: GO.** The full-suite result matches the targeted recheck's predictions exactly — improved where fragmentation was diagnosed as the cause (q015), unchanged where it was diagnosed as not the cause (q005). That consistency between a pre-registered prediction and the full-suite outcome is itself evidence the diagnosis was correct, not just that the numbers moved in a favorable direction.

---

*End of chronological record as of July 21, 2026.*