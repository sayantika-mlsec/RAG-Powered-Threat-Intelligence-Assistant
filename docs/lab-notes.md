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

**q004 — root cause: not a rewrite problem, a gate-threshold tradeoff.** Widened-pool dense search puts the correct answer at rank 1/15, distance 0.3336 — one of calibration's own recorded "success" distances, sitting in the overlap zone the conservative threshold was built to exclude from trust. Deferred to rewrite, where it's then answered wrong. Not a new bug — the threshold's design explicitly accepted this tradeoff at calibration time. The exact size of the sacrificed-success group ("~4" vs. 5 implied elsewhere) has not been reconciled — parked, see Known Limitations.

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

## Known Limitations — July 18 (original entries; see `eval_pipeline.md` for the current consolidated list including #4, added July 19)

### 1. Chunk fragmentation at ingestion (T1140, scope beyond it unknown)

T1140.txt is split across two separate chunks — confirmed by direct query, not inferred. One carries the generic definition, the other the worked example. The split falls at a paragraph boundary, consistent with a chunking strategy that doesn't recognize an example as content that should stay attached to its preceding sentence.

**Effect.** A query needing the example specifically (q003) fails whenever only the definition chunk is retrieved, but recall@3 still reports success — recall is scored at the `technique_id` label level and can't see a technique's content is split across chunks.

**Scope not established.** T1140 is the only technique confirmed so far, found while investigating one query's failure, not a systematic sweep.

**Why not fixed now.** Requires an ingestion-time change or a recall-metric change; both larger than a same-day fix, out of scope for the boundary.

### 2. Confidence-gate threshold precision loss

`calibrate_confidence.py` computes a full-precision threshold but only ever prints it rounded to 4 decimals, with a human hand-copying the value into `config.py`. No code path writes the exact value programmatically.

**Observed effect.** On q003, the best candidate sits at distance 0.3146 to 4-decimal precision — apparently tied with the stored threshold. `is_confident()`'s comparison is written correctly and would trust a genuine tie; in production this row still fell through to rewrite, consistent with the stored threshold being a rounded-down truncation. Not confirmed via direct `repr()`-level inspection — best available explanation, not proven.

**Why it matters beyond q003.** Systematic risk: any query whose top-1 distance lands within rounding distance of the true threshold is subject to the same silent misclassification, in either direction.

**Why not fixed now.** Low-risk, mechanical fix; not because it's difficult.

### 3. Overlap-zone success count, unreconciled (parked)

The Jul 14 q004 write-up refers to "~4" sacrificed successes; direct arithmetic from the same section's own numbers implies 5. Not reconciled — could be an approximation in original wording, or a genuine calibration-time vs. production-time search difference (the `both`-route caveat already named in `confidence_gate.py`'s docstring). Parked deliberately; the original success/failure distance lists would resolve this if revisited.

---

*End of chronological record as of July 19, 2026. Entries added July 19 (doc split, terminology table, resubstitution disclosure) are cross-referenced above and consolidated in `eval_pipeline.md`.*