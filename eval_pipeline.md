# RAG Evaluation Pipeline — Current State

*Last updated: July 19, 2026. This document reflects the current, validated state of the Tiered RAG evaluation pipeline. Full build history, dated findings, and superseded/retracted claims live in `docs/lab-notes.md` — this file states where things stand, not how they got there.*

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

**Faithfulness scores are directional, not precisely comparable across arms.** Repeated observation (q013 flipping 1→5 on identical retrieval and identical refusal behavior — judge variance on an unchanged input) shows single-run judge scores carry enough variance that a 0.5–1.0 point delta between arms should be read as noise unless independently corroborated. (q003 showed a similar-looking 5→1 flip but was root-caused to a genuine retrieval-composition difference between arms, not judge variance — see Known Limitations #1 and #6 — and is excluded from this claim.) Means are reported below for completeness but should not be treated as precise deltas. Full k=3 judge replication on boundary rows (q013, q015) is a documented upgrade path, not yet executed (see Deferred Work).

---

## Current Numbers (as of July 19, 2026)

### Retrieval

| Arm | Precision@3 | Recall@3 |
|---|---|---|
| Blind | 0.2444 | 0.5667 |
| Routed, no fixes | 0.2444 | 0.5667 |
| Routed + fixes (ungated) | 0.4667 | 0.7556 |
| **Routed + fixes + Confidence Gate** ¹ | **0.4778** | **0.8222** |

¹ *13 of these 15 queries' retrieval paths were determined by the Confidence Gate, calibrated on those same 13 queries — a resubstitution estimate for the gate's trust/no-trust decisions, not a held-out evaluation of them. The remaining 2 queries resolve via deterministic exact-match and never touch the gate. The true out-of-sample false-positive rate for the gate is unmeasured. See Known Limitations, #4.*

Zero misroutes recorded on every routing run to date (baseline 15-query set and the 8-query stress test's 6 unambiguous cases).

### Faithfulness

| Arm | Mean | N Eligible | N Ineligible |
|---|---|---|---|
| Blind | 4.444 | 9 | 6 |
| Routed (pre-fix retrieval, stale) | 5.000 | 9 | 6 |
| **Routed + fixes + Confidence Gate** | **4.429** | **14** | **1** |

Read directionally only (see Metrics note above). The single ineligible row under the current arm is q004 — consistent with its known-limitation diagnosis (dense search already correct, deferred to rewrite by the conservative threshold, rewrite answers it wrong).

### Tier Distribution & Cost/Latency (Routed Arm)

13 flash / 7 pro (35% pro). Two boundary-adjacent queries (q002, q013) confirmed stable across repeated runs post-`temperature=0.0` fix. A third boundary-adjacent query, q003, was later found **not** stable under the same conditions on the gated arm (flash → pro → pro across 3 repeats) — see Known Limitations #6. The 35% aggregate figure itself is not affected by this (it comes from a single full-suite capture, not from the repeated-run stability checks), but q003's individual tier assignment should be treated as unconfirmed rather than settled.

| Tier | N | Avg latency | Avg cost | Total cost |
|---|---|---|---|---|
| Flash | 9 | 2.74s | $0.00105 | $0.00941 |
| Pro | 7 | 11.37s | $0.01243 | $0.08703 |

Tiering saves 22.8% vs. all-Pro, costs 210% more than all-Flash — real but modest savings, since 44% of usable queries still land on the ~12x-more-expensive tier. Pricing is Developer API, not confirmed against actual Vertex billing.

### Routing Stress Test (8 additional queries, not retrieval-scored)

0 misroutes on 6 unambiguous cases including 3 deliberately cross-corpus ones. 1 misroute (q024 — KEV-style concept phrased in ATT&CK-style language, routed MITRE-only), reproducing the vocabulary-mismatch failure mode at the routing layer, not just retrieval. 1 defensible ambiguous case (q025, "tell me about ransomware" → skipped; `both` would also have been acceptable).

---

## Architecture (Current Pipeline)

1. **Route** — `route_query()` picks a corpus (`mitre_only` / `kev_only` / `both` / `skip`) and a model tier (`flash` / `pro`) in one structured-output call. `temperature=0.0` pinned.
2. **Retrieve** — `retrieve_for_route()`: exact-match ID lookup first (deterministic) → Confidence Gate checks dense-search distance against a calibrated threshold → trusted directly if confident, else falls through to corpus-tagged query rewrite with guaranteed-slot cross-corpus merge.
3. **Generate** — `ThreatAnalyzer.generate_answer()` dispatches to the tier-selected model (`gemini-2.5-flash` / `gemini-2.5-pro`), strict grounding instruction (context-only, exact refusal string on any gap).
4. **Score** — retrieval scored by precision/recall@3 against expected IDs; faithfulness scored by LLM-judge against retrieved context (not gold). All runs logged to MLflow with fail-loud reconciliation invariants and `recall_run_id` lineage tying faithfulness to the exact retrieval run.

---

## Known Limitations

**1. Chunk fragmentation at ingestion.** T1140 is split across two chunks (definition vs. worked example) at what looks like a paragraph boundary. Recall scores at the `technique_id` label level and can't see that a technique's answer may live in only one of several chunks. Scope beyond T1140 not established — found via one query's failure, not a systematic sweep. Not fixed: requires an ingestion-time or recall-metric change.

**2. Confidence-gate threshold precision loss.** `calibrate_confidence.py` computes a full-precision threshold but only ever prints it rounded to 4 decimals for manual transcription into `config.py`. On q003, a nominally-tied distance (0.3146) lost the confidence comparison, consistent with the stored value being a rounded truncation — not confirmed via direct `repr()` inspection. Low-risk, mechanical fix; not because it's difficult.

**3. Overlap-zone success count, unreconciled.** Two different counts exist in the history for how many calibration successes the conservative threshold sacrifices (~4 vs. 5, implied by different arithmetic in different sections). Not reconciled; parked.

**4. Confidence-gate calibration is resubstitution, not held-out validation.** The threshold was calibrated on the same 13 queries later used to determine the gated-arm's retrieval path for those same 13 of 15 scored rows. "Zero observed false positives" is a property of this exact set by construction, not a validated generalization bound. The true out-of-sample false-positive rate is unmeasured — likely optimistic relative to production. A single holdout split or leave-one-out cross-validation would resolve this; deferred given the boundary and n=13's already-limited statistical power.

**5. Faithfulness judge is lenient toward wrongful refusal.** An answer with no claims trivially passes the grounding check. Confirmed on q013 (blind vs. routed: same refusal, scores 1 then 5) and q015 (proof case). This is why faithfulness deltas are read directionally (see Metrics) and why a separate correctness-vs-gold metric is deferred but justified by data, not yet built.

**6. Tier classification instability, q003 (gated arm), unresolved.** Three repeated runs of q003 on the gated arm — identical query, identical route, identical retrieved chunks — classified tier as flash, then pro, then pro. This conflicts with the "confirmed stable across repeated runs" language used elsewhere for post-`temperature=0.0` classifications (q002, q013): the stability check behind that language was run against a different capture of q003 and was not re-verified on this specific gated-arm repeat (see `docs/lab-notes.md`, Jul 17/18 entries). Not root-caused. Not fixed: noted here so the current-state numbers don't overstate certainty on q003's individual tier assignment. Does not affect the 35% aggregate pro-rate, which comes from a separate single capture run.

---

## Deferred Work (not scheduled, not forgotten)

- Correctness-vs-gold metric (separate from faithfulness), to resolve the judge's refusal-blindness.
- Held-out or LOOCV validation of the confidence threshold (Limitation 4).
- Full-precision threshold write-through, removing manual transcription (Limitation 2).
- Judge k=3 replication on boundary rows (q013, q015) to quantify variance directly, as an alternative/upgrade to the current directional-only framing.
- Root-cause q003's gated-arm tier instability (Limitation 6) — currently only two repeated-run observations exist, in different sessions, not directly comparable.
- Tiered faithfulness re-score (faithfulness has not yet been scored against tier-tagged generation specifically).
- Eval-set growth past 20 queries — repeatedly named as the resolution to several small-n caveats above, not yet scheduled.

## See Also

Full chronological build log, dated findings, retracted claims, and the reasoning behind each fix: `docs/lab-notes.md`.