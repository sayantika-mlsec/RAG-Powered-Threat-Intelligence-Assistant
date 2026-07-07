# ADR 001: Agentic Routing over Single-Pass RAG

## Status
Accepted

## Context
The RAG assistant originally retrieved from both the MITRE ATT&CK and CISA KEV ChromaDB collections on every query, regardless of relevance. This "blind" retrieval wastes a retrieval pass on the wrong corpus for corpus-specific queries, and returns irrelevant chunks for queries that need no retrieval at all (greetings, meta-questions).

An agentic routing layer was added: a Gemini function-calling step classifies each query into one of four routes (MITRE-only, KEV-only, both, skip) before retrieval runs.

An eval was run A/B (blind vs. routed) on the same 20-query eval set, measuring Precision@3, Recall@3, and faithfulness.

## Decision
Keep agentic routing as the retrieval front-end, despite Precision@3 land ingidentical between arms (0.2444, delta 0.0000, 0 misroutes).

## Alternatives Considered
- **Hardcoded keyword routing (if/else on query terms).** Rejected: brittle against paraphrasing and out-of-vocabulary queries; doesn't generalize past the exact keywords anticipated at write time.

- **Always retrieve both collections (the blind baseline).** Rejected as the long-term design, kept as the comparison baseline. Simpler, but pays a retrieval-and-context-window cost on every query and has no skip path.

## Why the decision holds despite a flat precision delta
1. **The eval set didn't exercise where routing pays off — confirmed by a follow-up stress test.** Precision@3 is flat because misroutes were already at zero on the 20-query eval set. A follow-up stress test (8 deliberately adversarial queries, cross-corpus phrasing with no explicit IDs) held at 0 misroutes on all 6 unambiguous cases, including three cross-corpus bridges the router had to infer without any CVE/technique ID present. This confirms the flat delta reflects eval-set coverage, not routing being inert.
2. **The skip path prevents a failure mode precision can't see.** Greeting and meta-queries retrieved irrelevant chunks under blind retrieval, which risked the model citing them anyway. Routing's skip path removes that failure mode structurally, before generation ever runs.
3. **Honest cost, stated plainly.** Routing adds one Gemini function-calling round trip per query — extra latency and cost — for zero measured Precision@3 gain on this eval set. That's a real tradeoff, not one routing "wins" on the numbers currently available.

Note: the faithfulness arm showed routed 5.000 vs. blind 4.444 (+0.556), but this is **not** used as a ground for this decision. 2 of 9 eligible rows are refusals scoring 5 for making no claims, and retrieval was flat — there's no retrieval-side mechanism to explain a genuine faithfulness gain. Treated as noise pending a correctness-vs-gold metric (parked, tracked separately).

## Consequences
- **Positive:** skip-path eliminates a class of citation risk on non-retrieval queries; routing logic is now rea function-calling, not brittle keyword matching, so it should generalize better as query variety grows.
- **Negative:** added latency and API cost per query; a new failure mode (misrouting) is introduced even though it measured zero on this eval set; routing correctness now depends on Gemini's function-calling reliability, an external dependency. A stress test also reproduced the known vocabulary-mismatch failure at the routing layer itself (a KEV-appropriate query phrased in ATT&CK-style language routed to the wrong single corpus) — this is the same systemic issue already tracked in retrieval, now known to affect routing too, not a new problem.
- **Follow-up:** hybrid-retrieval work (addressing the actual precision ceiling — exact-identifier and vocabulary-mismatch failures) remains the higher-leverage fix and is tracked as a known limitation, not built yet.