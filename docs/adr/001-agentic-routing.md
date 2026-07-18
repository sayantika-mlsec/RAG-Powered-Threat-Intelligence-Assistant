# ADR 001: Agentic Routing over Single-Pass RAG

## Status
Accepted

## Context
The RAG assistant originally retrieved from both the MITRE ATT&CK and CISA KEV ChromaDB collections on every query, regardless of relevance. This "blind" retrieval wastes a retrieval pass on the wrong corpus for corpus-specific queries, and returns irrelevant chunks for queries that need no retrieval at all (greetings, meta-questions).

An agentic routing layer was added: a Gemini function-calling step classifies each query into one of four routes (MITRE-only, KEV-only, both, skip) before retrieval runs.

An eval was run A/B (blind vs. routed) on the same 20-query eval set, measuring Precision@3, Recall@3, and faithfulness.

## Decision
Keep agentic routing as the retrieval front-end, despite Precision@3 and Recall@3 being identical between arms (0.2444 / 0.5667, delta 0.0000 on both, 0 misroutes).

## Alternatives Considered
- **Hardcoded keyword routing (if/else on query terms).** Rejected: brittle against paraphrasing and out-of-vocabulary queries; doesn't generalize past the exact keywords anticipated at write time.

- **Always retrieve both collections (the blind baseline).** Rejected as the long-term design, kept as the comparison baseline. Simpler, but pays a retrieval-and-context-window cost on every query and has no skip path.

## Why the decision holds despite a flat precision delta

1. **The eval set didn't exercise where routing pays off — confirmed by a follow-up stress test.** Precision@3 is flat because misroutes were already at zero on the 20-query eval set. A follow-up stress test (8 deliberately adversarial queries, cross-corpus phrasing with no explicit IDs) held at 0 misroutes on all 6 unambiguous cases, including three cross-corpus bridges the router had to infer without any CVE/technique ID present. This confirms the flat delta reflects eval-set coverage, not routing being inert.

2. **Direct verification shows routing changes retrieval composition even when the aggregate metric doesn't move.** A row-by-row comparison of `retrieved_ids` between arms (`compare_retrieval_arms.py`) found 14 of 15 rows byte-identical, but q013 was not: both arms retrieved the two correct CVEs in the top two slots, but the third slot differed — blind pulled in a structurally irrelevant MITRE technique, routed swapped in a different, still-wrong CVE. Both score equally as "not expected," so precision and recall don't register the change, but the chunk actually returned is not the same chunk. This is direct, row-level evidence that routing has a real effect on what reaches the generator, independent of whether the aggregate score reflects it — a stronger form of proof than the stress-test argument in (1) alone.

3. **The skip path prevents a failure mode precision can't see.** Greeting and meta-queries retrieved irrelevant chunks under blind retrieval, which risked the model citing them anyway. Routing's skip path removes that failure mode structurally, before generation ever runs.

4. **Honest cost, stated plainly.** Routing adds one Gemini function-calling round trip per query — extra latency and cost — for zero measured Precision@3 gain on this eval set. That's a real tradeoff, not one routing "wins" on the numbers currently available.

Note: the faithfulness arm showed routed 5.000 vs. blind 4.444 (+0.556), but this is **not** used as grounds for this decision. 2 of 9 eligible rows are refusals scoring 5 for making no claims, and retrieval was flat — there's no retrieval-side mechanism to explain a genuine faithfulness gain. Treated as noise pending a correctness-vs-gold metric, consistent with the directional-only treatment of faithfulness deltas documented project-wide (see `eval_pipeline.md`, Metrics and Known Limitations #5) — the same judge-variance pattern, not re-derived independently here.

## Consequences
- **Positive:** the skip path eliminates a class of citation risk on non-retrieval queries; routing logic is now real function-calling, not brittle keyword matching, so it should generalize better as query variety grows; row-level composition changes (point 2) suggest routing effects will become visible in aggregate metrics as the eval set grows past cases where blind retrieval already happens to rank correctly.
- **Negative:** added latency and API cost per query; a new failure mode (misrouting) is introduced even though it measured zero on this eval set; routing correctness now depends on Gemini's function-calling reliability, an external dependency. The stress test also reproduced the known vocabulary-mismatch failure at the routing layer itself (a KEV-appropriate query phrased in ATT&CK-style language routed to the wrong single corpus, q024) — the same systemic issue already tracked in retrieval, now known to affect routing too, not a new problem.
- **Follow-up:** the precision ceiling this ADR left open has since been partially addressed, as a separate engineering effort layered on top of routing rather than a change to the routing decision itself. Exact-match ID lookup, corpus-tagged query rewrite, and guaranteed-slot cross-corpus merge (Jul 13) raised precision@3 0.2444→0.4667 and recall@3 0.5667→0.7556 on top of routing. A confidence-gate threshold (Jul 14) — dense search trusted directly above a calibrated distance, falling through to rewrite otherwise — added stability against rewrite's run-to-run non-determinism and raised the numbers further to 0.4778/0.8222. Both are retrieval-layer fixes; routing's own precision/recall stayed flat throughout (see Decision section above — 0.2444/0.5667 identical between blind and routed-no-fixes). Still open: the confidence-gate threshold is calibrated and evaluated on the same query set (resubstitution, not held-out — see `eval_pipeline.md`, Known Limitations #4). The exact-match mechanism already resolves the CVE/technique-ID-pattern cases originally flagged as the exact-identifier gap (q011, q012); the narrower residual — name-based vulnerability references outside that ID pattern (e.g. "Log4Shell," "MOVEit," both present only in the unscored routing stress test, not yet retrieval-tested) and the vocabulary-mismatch cases (q008, q010) — is what a hybrid lexical (BM25) layer and cross-encoder reranking remain unbuilt for. See `docs/lab-notes.md`, Baseline Failure Analysis summary table and the "Known Limitation — Residual Dense-Retrieval Ranking Miss (q015, q016)" entry — not `eval_pipeline.md`'s numbered Known Limitations, which cover different issues (chunk fragmentation, gate precision, overlap-zone count).