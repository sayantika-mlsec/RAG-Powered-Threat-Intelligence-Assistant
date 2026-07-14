# RAG Evaluation Pipeline

How the Threat Intelligence RAG assistant is evaluated, why these metrics were chosen, and what the baseline measurement found. The pipeline scores the single-pass (blind-retrieval) RAG; the agentic routing layer is evaluated against this same harness and compared to these numbers (see **Routing arm results**).

## Why evaluate at all

A RAG demo answers questions. A RAG *system* has to prove its answers are trustworthy under questioning — which retrieved chunks were right, which answers were actually grounded in them, and where it fails. Without measurement, "it works" is an assertion. The pipeline replaces that assertion with numbers and, more importantly, with a documented set of failure modes that say *where* it breaks and *why*.

The evaluation is split into two independently meaningful halves:

- **Retrieval** — did the right chunks come back in the top K? (precision@K, recall@K)
- **Faithfulness** — given the chunks that came back, is the generated answer actually grounded in them? (LLM-as-judge, 1–5 rubric)

These are deliberately separate. Retrieval can succeed while generation hallucinates; generation can be faithful to context that was itself wrong. Each half isolates one stage so a failure points at the responsible component.

## Eval-set construction

The eval set is 20 queries (`eval_set.json`), each with a stable `id` (`q001`–`q020`) used as the join key across every artifact in the pipeline. Each row carries the query, `expected_technique_ids` (the MITRE techniques that should be retrieved — empty for pure CVE-lookup rows, which are scored against the correct CVE entry instead), and a manually written gold-answer summary.

Gold answers are written **by hand** This is the load-bearing decision of the whole pipeline: an AI-generated gold answer is just another model output, so scoring against it measures agreement between two models, not correctness. Hand-written gold answers — verified against the actually-ingested chunks — are the only real ground truth available. (As the failure analysis shows, even hand-written golds can encode assumptions the system's prompt forbids — see q009 — which is itself a finding, not a flaw in the approach.)

The 20 rows split into a 15-row Group A (the reconciliation total used throughout) plus 5 rows with no expected IDs (q014, q017–q020) — skip-path and no-retrieval probes, not scored against retrieval — which are never retrieval-scored and appear in the artifacts as `n_gated_out`. Thin technique categories and new queries (`q021+`) are grown alongside the routing work.

## Retrieval metrics — precision@K and recall@K

**Precision@K** asks: of the K chunks retrieved, how many were relevant?
**Recall@K** asks: of all the relevant chunks that exist, how many did we get in the top K? Both are measured at **K=3** against `expected_technique_ids`.

Recall is the metric that gates the rest of the pipeline. A row with recall@3 = 0 — the right chunk never came back - is**ineligible for faithfulness scoring**, because faithfulness measures answer-vs-retrieved-context, and if retrieval failed there is no correct context to be faithful to. Scoring those rows would measure the generator's behaviour on garbage input, which is a different question. They are faithfulness-ineligible, counted, and analysed separately (see failure analysis). Note: this is distinct from the retrieval artifact's `n_gated_out` field, which counts rows with no expected IDs that are never retrieval-scored at all.

Per-row recall is emitted as an MLflow artifact with fail-loud invariants: a K-pin guard (the K used must match the K claimed), a corpus stamp (chunk count recorded, catching partial/duplicate ingest; content edits at constant count are not detected — corpus version is guaranteed by run lineage), and a reconciliation check. The eligibility decision for every downstream stage is *read from this artifact by run id*, never recomputed — so faithfulness is provably tied to the exact retrieval run that produced eligibility.

## Faithfulness metric — LLM-as-judge

Faithfulness asks one question: do the generated answer's claims appear in the retrieved context? It is scored by **Gemini 2.5 Flash as judge**, on a 1–5 rubric (5 = every claim traces to context; 1 = central claim contradicts context or is fabricated). The judge sees only the answer and the context — **gold answers are not an input.** Faithfulness is grounding, not correctness; the two are separated on purpose.

**Why LLM-as-judge over cosine similarity.** A cheaper alternative is cosine similarity between answer and context embeddings. It was rejected as the default (documented as a fallback if cost/latency forces it) because similarity measures *topical overlap*, not *entailment* — an answer can be highly similar to context it actually contradicts. A rubric-driven judge reasons about whether each claim is supported, which is the property faithfulness is supposed to capture.

**Judge implementation note.** The judge runs on the new `google.genai` SDK specifically to disable model thinking (`ThinkingConfig(thinking_budget=0)`): `gemini-2.5-flash` thinks by default, and the thinking tokens consumed the output budget and truncated the JSON reply. The old `google.generativeai` SDK has no `ThinkingConfig`, which forced the judge — and only the judge — onto the new SDK. Generation and ingest remain on the old SDK; the two coexist intentionally and transitionally. (This two-SDK decision is an ADR candidate.)

**Known limitation.** The judge is structurally blind to wrongful refusal — an answer that makes no claims trivially passes the grounding check. This is detailed in the failure analysis with q015 as the proof case, and is the concrete reason a separate correctness-against-gold metric is necessary (deferred, but justified by data). As of the gated re-score (below), this limitation is now confirmed **bidirectional**: the judge is inconsistent in both directions on refusal-shaped answers, not just biased toward leniency.

## Logging contract

The faithfulness score is **never logged bare.** It travels with N (eligible row count), the gated-out count, and `recall_run_id` lineage, in a single MLflow run. Reconciliation (`N_eligible + N_ineligible == 15` — within the scored set) runs *before* any scoring — if the artifacts disagree about the eval set, the run stops rather than scoring against an inconsistent denominator. The empty-eligible case (N=0) logs null rather than dividing by zero. Bad judge output raises rather than coercing to a middle value.

## Baseline numbers (single-pass RAG)

| Metric | Value |
|---|---|
| Precision@3 | 0.2444 |
| Recall@3 | 0.5667 |
| Faithfulness mean | 4.444 (N=9 eligible) |
| Faithfulness-ineligible (recall@3 = 0) | 6 |
| Group A total | 15 (reconciles: 9 + 6) |
| Corpus | chunk_count = 2140 |


The faithfulness mean of 4.444 is read with its caveat: it is computed over the 9 rows that passed the recall gate, and it is inflated by refusals that score high for making no claims (see judge limitation). It is a baseline to beat, not a headline to celebrate.

## Baseline failure analysis

The single-pass RAG baseline produces two recorded runs: retrieval (precision
0.2444 / recall 0.5667 @ K=3) and faithfulness (mean 4.444 over N=9 eligible, 6 gated out, reconciles to 15). The
mean alone is not the finding. The failures behind it fall into two classes — **generation** (the right context was retrieved, the answer still failed) and **retrieval** (the right chunk never came back, so the generator never had a chance) — plus a structural limitation in the faithfulness judge itself.

### Generation failures (eligible rows — context was retrieved)

All three eligible anomalies returned the *same* surface answer — "I do not have sufficient information in the provided threat intelligence." They are not the same failure. The generator's system instruction enforces strict grounding: answer only from context, *do not use prior knowledge*, and on any gap respond *exactly* with the refusal string. That all-or-nothing contract interacts with how the eval queries and gold answers were written.

- **q013 — over-refusal (prompt defect).** The query asks which Microsoft GDI privilege-escalation flaw is in the exploited      
catalog. The CVE (CVE-2017-0005) is present in the retrieved context. But "exploited catalog" asks about *KEV provenance* — the fact that the entry originates from CISA's known-exploited list — which no chunk states in its text. Under the all-or-nothing rule, the model refuses the whole query rather than answering the part it can support. The judge correctly scored this 1: the refusal contradicts a context that contained the answer.

- **q009 — legitimate refusal (eval-set defect).** The query asks for the Microsoft Windows system built on Web-Based Enterprise Management standards.The answer is WMI (T1047), and the T1047 chunk was retrieved. But the context does not contain the string "WBEM" or "Web-Based Enterprise Management" (verified). Bridging "WBEM" to WMI requires real-world domain knowledge, which the prompt explicitly forbids. The model obeyed its instructions. The defect is in the gold answer, which assumed a domain bridge the prompt bans — not in the generator.

- **q015 — mixed (retrieval miss + over-refusal).** The query has two clauses: is the Cisco IP Phones RCE flaw actively exploited, and what technique covers exploiting an internet-facing service (T1210). The CVE chunk (CVE-2020-3161) was retrieved, but "actively exploited" is the same KEV-provenance gap as q013, and the T1210 chunk was not retrieved at all. Part of this query is genuinely unanswerable from the retrieved context; part is over-refusal. Not a clean single cause.

The through-line: **strict no-prior-knowledge grounding trades recall for faithfulness.** The same rule that prevents hallucination forces refusal whenever a query's framing isn't literally present in chunk text. q013 (over-refusal) and q009 (correct refusal on a bad gold) are the two poles of that tradeoff.

### Judge limitation — blindness to wrongful refusal

The faithfulness judge is mechanically sound but has a structural blind spot, and q015 is the proof. An answer that makes no claims trivially passes a grounding check — there is nothing to contradict the context — so the judge scores refusals high regardless of whether the refusal was correct. It scored q009 and q015 a faithful 5, and caught q013's wrongful refusal only because it happened to reason "contradicts" rather than "no claims." On q015 the judge even reasoned the context was *thin* when the CVE chunk was in fact present — wrong about the context, yet a defensible score, because under the rubric a no-claims answer
passes either way.

This is not a judge bug to fix. It is an inherent property of grounding-only scoring: **faithfulness penalizes hallucination but is blind to under-answering. A model that refuses everything scores near-perfect faithfulness.** This is the concrete rows-behind-it evidence that faithfulness ≠ correctness, and the reason a separate correctness-against-gold metric is necessary (deferred, but now justified by data rather than assertion). (The routed run later confirmed this: the same q013 refusal scored 5 — see Routing arm results. The gated re-score confirms it a second time, independently, and adds a mirror-image case — see Gated Faithfulness Re-Score below.)

### Retrieval failures (ineligible rows — recall@3 = 0)

Six rows never returned the correct chunk in the top 3, so they were gated out of faithfulness scoring by design. They split into three sub-modes by *why* dense retrieval missed.

- **Exact-identifier — q011, q012.** Both queries are dominated by a CVE number ("What is CVE-2012-4681...", "What is CVE-2016-3715..."). Their `expected_technique_ids` are empty — these are CVE lookups, not technique lookups, so recall is measured against whether the right *CVE entry* came back, not a technique. A CVE ID is a near-opaque token: `all-MiniLM-L6-v2` embeds semantic meaning, and the ID carries almost none, so the query embedding lands in a generic "some vulnerability" region and the exact entry doesn't surface. With no technique vocabulary to fall back on, these are the cleanest possible illustration of dense retrieval failing on exact identifiers. *Fix: hybrid (dense + lexical/metadata-exact) retrieval — ADR candidate.*

- **Multi-hop / compositional — q005, q016.** q005 expects three techniques (T1078, T1098, T1111 — defeat MFA, valid accounts, modify account settings); q016 expects a CVE plus T1210. A single query embedding averages across the clauses and matches none of them strongly enough to retrieve every required chunk in the top 3. The corpus has the answers; one embedding can't retrieve a
multi-part answer. *Fix: query decomposition — a separate operation from collection routing, which selects a corpus but does not split a multi-clause query into sub-queries.*

- **Vocabulary mismatch — q008, q010.** The answers exist (T1041 Exfiltration Over C2 Channel; T1105 Ingress Tool Transfer) but the queries are written in plain English with zero MITRE vocabulary ("sneaking data out through the same connection they use to control their malware"). The chunks are written in technical terms; the embeddings don't bridge the gap. *Fix: query expansion or a reranker; partly an inherent embedding-model limit.*

### Summary

| Row | Class | Sub-mode | Fix lands in |
|---|---|---|---|
| q009 | Generation | Legitimate refusal (bad gold) | Eval-set revision |
| q013 | Generation | Over-refusal (strict prompt) | Prompt loosening |
| q015 | Mixed | Over-refusal + retrieval miss | Prompt + routing |
| q005 | Retrieval | Multi-hop (T1078/T1098/T1111) | Query decomposition |
| q016 | Retrieval | Multi-hop (CVE + T1210) | Query decomposition |
| q008 | Retrieval | Vocabulary mismatch (T1041) | Query expansion / reranker |
| q010 | Retrieval | Vocabulary mismatch (T1105) | Query expansion / reranker |
| q011 | Retrieval | Exact-identifier (CVE, no technique) | Hybrid retrieval (ADR) |
| q012 | Retrieval | Exact-identifier (CVE, no technique) | Hybrid retrieval (ADR) |

No fixes are applied in this issue — it characterizes failures. Generation over-refusal and the judge blind spot inform the prompt and metric work; multi-hop and vocabulary-mismatch retrieval inform the routing layer; exact-identifier retrieval is the hybrid-retrieval ADR candidate.

## Routing arm results

The agentic routing layer was evaluated against the same K=3 harness and compared to the blind baseline. `use_routing=True` sends each query through `route_query`, which picks a corpus (`mitre_only`, `kev_only`, `both`, or `skip`); retrieval is then filtered to the routed corpus instead of querying the whole store.

### Retrieval — routed vs blind

| Metric | Blind baseline | Routed | Delta |
|---|---|---|---|
| Precision@3 | 0.2444 | 0.2444 | 0.0000 |
| Recall@3 | 0.5667 | 0.5667 | 0.0000 |
| n_misroutes | — | 0 | — |
| Reconciliation | OK | OK (scored + gated == total) | — |

**The result is flat, and that is the finding.** Routing changed neither precision nor recall at K=3, with zero misroutes. Read together, those two facts localize what routing does and does not do:

- **Routing did its job.** `n_misroutes = 0` means every scored query reached the correct corpus — no query was sent to the wrong store or wrongly skipped. Corpus selection is correct.
- **Correct corpus selection did not move top-3.** The store imbalance the routing layer was built to counter (540 MITRE chunks vs 1600 KEV; per-corpus counts are not currently stamped in the artifact — only the combined 2140; per-corpus stamping is a scheduled fix) does distort deeper ranks — a MITRE query under `both` competes against 1600 KEV chunks — but for this eval set, recall was identical row-by-row between arms — consistent with the correct chunks already ranking in the top 3 under blind retrieval. (The per-row artifact does not yet store `retrieved_ids`, so top-3 identity is inferred from recall parity, not demonstrated chunk-by-chunk; adding `retrieved_ids` is a scheduled artifact upgrade.) Filtering out wrong-corpus chunks that sat at rank 4+ changed nothing about which three chunks won the top-3 slots. Routing removed noise that K=3 never saw.

### Per-category breakdown

| Corpus | Difficulty | Precision@3 | Recall@3 |
|---|---|---|---|
| cross | hard | 0.1667 | 0.2500 |
| kev | easy | 0.0000 | 0.0000 |
| kev | medium | 0.6667 | 1.0000 |
| mitre | easy | 0.0000 | 0.0000 |
| mitre | hard | 0.3333 | 0.7500 |
| mitre | medium | 0.2667 | 0.8000 |

The two `easy` buckets sit at 0.0000/0.0000 for both corpora — and with `n_misroutes = 0`, this is **not** a routing failure. These queries were correctly routed, hit the correct store, and the right chunks still did not come back in the top 3. That places the residual failure squarely at the retrieval layer, not the routing layer — the exact-identifier and vocabulary-mismatch modes already characterized in the baseline failure analysis (q008/q010/q011/q012). Routing cannot fix them because they were never routing problems.

### What this run establishes

Agentic routing held precision constant at K=3 with zero misroutes; the residual precision ceiling is a retrieval-layer problem, not a corpus-selection one. This is the measured evidence — not a prediction — behind the hybrid-retrieval ADR: routing is the right tool for corpus selection and the wrong tool for exact-identifier and vocab-mismatch lookup, and the data now shows exactly that separation. A flat precision delta with a clean misroute count is a stronger localization of the next bottleneck than a precision bump would have been.

### Routed Faithfulness Results

| Arm    | Faithfulness Mean | N Eligible | N Ineligible |
|--------|-------------------|------------|--------------|
| Blind  | 4.444              | 9          | 6            |
| Routed | 5.000              | 9          | 6            |

The delta decomposes exactly: sums of 40 vs 45 over the same 9 rows. q013 — the wrongful refusal the blind-arm judge scored 1 ("contradicts context") — scored 5 on this run ("no claims to contradict"): +4 of the 5 points from one row, on identical retrieval and identical refusal behavior. The remaining +1 is a single row moving 4→5. The delta is judge variance concentrated on ~2 rows, not a system change — and the q013 flip is direct confirmation of the blind-arm prediction that the judge caught q013 "only because it happened to reason 'contradicts.'"

Delta: +0.556. Read with the standing caveat on this metric: faithfulness scores grounding, not correctness, and a refusal that makes no claims trivially passes. Two of the nine eligible rows this run (q013, q015) are refusals scored 5 for that reason — q015 is the same proof case already documented on the blind arm. Excluding both, 7/7 remaining rows scored 5 on genuinely grounded, substantive answers.

**Reading the delta honestly:** with N=9 and 2 of those being refusal-inflated, this is not strong evidence that routing improved
faithfulness — retrieval was flat (delta 0.0000) between arms, so there's no retrieval-side mechanism that would explain a real faithfulness gain. The safer read: faithfulness on the *retrieved-and-answered* subset is consistently high in both arms; the delta itself is judge-scoring variance concentrated on refusal rows, not a routing-driven improvement (see decomposition above). A correctness-vs-gold metric (fix-ordering: retrieval → correctness → prompt) would resolve this ambiguity — noted as future work.

## Routing Stress Test
 
The original 20-query eval set showed 0 misroutes, but wasn't designed to stress the router — it didn't contain deliberately ambiguous or cross-corpus phrasing. Ran 8 additional queries (`stress_test_routes.py`) specifically targeting: cross-corpus bridging with no explicit IDs, bare identifiers (sanity check), the skip path, and vocabulary-mismatch phrasing. These 8 queries probe routing decisions only — they were not retrieval-scored, so they have no recall numbers and do not affect the metrics above. Expected values match the `Route` enum's `.value` strings; q025 accepts either of two routes (pipe-separated) because it is ambiguous by design.
 
| Query ID | Query | Expected Route | Actual Route | Misroute? |
|---|---|---|---|---|
| q021 | What technique does Log4Shell map to, and is it actively exploited? | both | both | no |
| q022 | Which actively exploited vulnerabilities involve lateral movement? | both | both | no |
| q023 | Is the MOVEit vulnerability linked to a known credential access technique? | both | both | no |
| q024 | How do attackers get in through unpatched internet-facing software? | kev_only | mitre_only | yes |
| q025 | Tell me about ransomware. | skip\|both | skip | no |
| q026 | T1190 | mitre_only | mitre_only | no |
| q027 | CVE-2021-34473 | kev_only | kev_only | no |
| q028 | What did we just talk about? | skip | skip | no |
 
**Result: 0 misroutes on the 6 clean cases (q021–q023, q026–q028)**, including all three deliberately cross-corpus ones. The router correctly bridged CVE-name-to-technique and concept-to-both-corpora reasoning with no explicit IDs in the query text — a harder case than anything in the original 20-query set, and it held.
 
**q024 is the one recorded misroute, and it reproduces the known vocabulary-mismatch failure mode at the routing layer, not just retrieval.** The query describes a KEV-style concept (exploiting unpatched internet-facing software) but was routed MITRE-only, because the phrasing ("attackers get in") pattern-matched ATT&CK-style language. This suggests the vocabulary-mismatch problem documented in retrieval (q008/q010) is systemic to how the corpora are described and labeled, not isolated to the retrieval layer.
 
**q025 routed to skip**, judging "tell me about ransomware" too broad to answer from either corpus. Both `skip` and `both` are accepted as correct for this row: skip is the defensible conservative call, though arguably `both` would have surfaced useful general context — a design tradeoff worth naming, not a bug.

## Retrieval Precision-Ceiling Fixes — Exact-Match, Query Rewrite, Cross-Corpus Merge - July 13

**Issue:** #17 — "Fix RAG retrieval precision ceiling: exact-ID matching, query rewriting, cross-corpus routing"

### What was fixed

Three mechanisms, added as a new `use_retrieval_fixes` arm (independent of`use_routing` — see updated `run_evaluation` docstring):

1. **Exact-match ID lookup** (`exact_id.py`) — regex-extracts all literal CVE and MITRE technique IDs from the raw query, bypasses the embedder entirely via a direct ChromaDB metadata filter. Fully deterministic.
2. **Corpus-tagged query rewrite** (`query_rewrite.py`) — single Gemini call splits a query into 1-4 sub-queries when it genuinely chains distinct actions, tags each with a predicted corpus (`mitre`/`kev`). Falls back to a single reworded query for non-ID, non-multi-hop queries.
3. **Guaranteed-slot cross-corpus merge** — each sub-query's own best result gets a guaranteed seat in the final top-K before remaining slots fill globally by distance. Fixes a real starvation bug where one sub-query's near-miss results could crowd out another sub-query's correct answer purely because raw distances aren't comparable across sub-queries searching different corpora.

### Results — confirmed

| Arm | Precision@3 | Recall@3 |
|---|---|---|
| Blind (pinned baseline) | 0.2444 | 0.5667 |
| Routed (no fixes) | 0.2444 | 0.5667 *(identical to blind — confirms 0 misroutes, routing alone doesn't move retrieval precision)* |
| Routed + fixes | **0.4667** | **0.7556** *(confirmed: category-weighted cross-check against per-row data matches exactly — 7.00/15 precision, 11.333/15 recall)* |

Category breakdown, routed + fixes:

| Category | Difficulty | Precision@3 | Recall@3 |
|---|---|---|---|
| cross | hard | 0.3333 | 0.5000 |
| kev | easy | 1.0000 | 1.0000 |
| kev | medium | 0.6667 | 1.0000 |
| mitre | easy | 0.3333 | 1.0000 |
| mitre | hard | 0.4167 | 0.5833 |
| mitre | medium | 0.3333 | 0.8000 |

Net: **+91% precision, +33% recall** vs. baseline, from retrieval-layer fixes alone — routing was already correct going in (0 misroutes on every run this session).

### Known Limitation 1 — Residual Dense-Retrieval Ranking Miss (q015, q016)

Both cross-corpus queries (route=`both`) still miss one half of their expected answer even after all fixes: q015's T1190 (MITRE side) and q016's CVE-2020-0796 (KEV side) are both confirmed present in their sub-query's own candidate pool at low rank (rank 13/15 and 5/15 respectively, via direct widened-pool inspection) — a genuine embedding-ranking limitation, not a bug in the merge or routing logic. Pool-widening (fetching more candidates per sub-query so a low-ranked correct answer could surface during merge) was evaluated and found insufficient on its own for q015 (T1190 still ranked 13th even with canonical MITRE phrasing);
not shipped.

**Candidate follow-ups (not in scope here):** cross-encoder re-ranking on a widened per-sub-query candidate pool before final merge; chunk-level rewording to align source text closer to expected query vocabulary; or a hybrid lexical layer (BM25).

### Known Limitation 2 — Rewrite Step Is Not Deterministic Across Runs

Across five full-suite and isolated runs of the identical pipeline against
identical queries, the rewrite step (`temperature=0.0`, not guaranteed
deterministic per Gemini's own floating-point behavior across GPU cores)
produced measurably different outcomes run-to-run:

- **q016**: three different outcomes across three runs on the SAME query — a full rescue (T1210 recovered via guaranteed-slot merge), a full miss (JSON truncation fallback, later root-caused to a missing `thinking_config=ThinkingConfig(thinking_budget=0)` setting and fixed — same fix pattern already applied to the faithfulness judge elsewhere in this project), and a different full miss (wrong sub-query split entirely, once truncation was fixed).
- **q001**: passed on plain dense search (routed, no fixes), then FAILED after the rewrite step incorrectly split a single-concept query into two sub-queries based on a coincidental keyword overlap ("command-and-control" read as a second technique rather than descriptive context). A targeted prompt rule was added to prevent this; q001 still failed on the subsequent run.
- **q004**: passed in every prior run, including the immediately preceding one, then regressed to a full miss the run immediately after an unrelated prompt-rule addition (parent-technique preference, added to fix q007) — confirming the rewrite prompt's rules interact with each other in ways not fully predictable from the ruleset alone.
- **q007**: failed initially (over-specific sub-technique selected instead of the parent technique the query asked for — matches this query's own `difficulty_note` in `eval_set.json`), fixed by an explicit "prefer parent technique" prompt rule, held on the one subsequent run.

**Root cause:** the rewrite step is an LLM call making a judgment (split-or-not, which corpus, which phrasing, which granularity) that is sensitive to exact wording variation between otherwise-identical calls. Four rounds of prompt patching during this session each fixed the specific failure being chased while occasionally introducing or failing to resolve others — diminishing returns consistent with prompt-only iteration having a real ceiling on this class of problem.

**Contrast worth noting:** the exact-match component (mechanism 1) is fully deterministic and was correct on every single run, no exceptions — the instability is isolated entirely to the LLM-driven rewrite step (mechanisms 2-3), not the pipeline as a whole.

## Fallback-Only Rewrite Restructuring (Confidence Gate) - July 14

**Motivation:** the Jul 13 routed_with_fixes arm improved precision/recall (0.2444→0.4667 / 0.5667→0.7556) but introduced a stability problem: rewrite ran unconditionally on every non-exact-match query, and at least one query (q001) that dense search already answered correctly got broken by rewrite splitting it on a keyword coincidence.

**Design:** `retrieve_for_route()` now tries dense search on the raw query *before* rewrite, and only falls through to rewrite if dense search isn't confident. "Confident" = the top-1 result's distance is at or below a threshold calibrated offline — there's no ground truth to check against at inference time (see `confidence_gate.py` module docstring for the full reasoning).

**Calibration methodology:**
- Ran dense-search-only over the eval set's 15 scored queries.
- Excluded 2 queries that resolve via exact-match in production (they never reach the gate) and 0 misroutes this run — 13 queries used.
- Split top-1 distances by whether the retrieved chunk's ID was in `expected_ids`: 8 successes (range 0.2328–0.4446), 5 failures
  (range 0.324–0.4464). The two distributions overlap substantially.
- Two threshold candidates: coverage-optimized (75th percentile of successes → 0.3775) vs. conservative (largest success distance below the smallest failure distance → 0.3146).
- **Chose the conservative threshold (0.3146).** The two error types aren't equal cost: a wrongly-trusted dense result is guaranteed wrong with no rewrite chance to correct it, while a correct dense result sent to rewrite unnecessarily is usually still right. At 0.3775, 3 of 5 observed failures would have been wrongly trusted; at 0.3146, zero are.
- Caveat: n=13 is small. "Zero observed false positives" is a property of this eval set, not a guarantee — revisit if/when the eval set grows past 30–50 queries.

**Results — 3 independent runs, frozen pipeline** (no prompt edits between runs, unlike Jul 13's four-rounds-of-patching session):

| Run | Path distribution | Precision@3 | Recall@3 |
|---|---|---|---|
| 1 | exact_match: 2, dense_confident: 3, rewrite: 10 | 0.4778 | 0.8222 |
| 2 | identical to run 1 | identical | identical |
| 3 | identical to run 1 | identical | identical |

All 15 rows — including all 10 `rewrite`-path rows — returned byte-identical `retrieved_ids` across all 3 runs. Verified deliberately: the first repeat was assumed to be a re-pasted/cached artifact rather than a real result until MLflow run IDs and timestamps confirmed all 3 were independent executions.

| Arm | Precision@3 | Recall@3 |
|---|---|---|
| Blind | 0.2444 | 0.5667 |
| Routed, no fixes | 0.2444 | 0.5667 |
| Routed + fixes (ungated, Jul 13) | 0.4667 | 0.7556 |
| **Routed + fixes + gate (Jul 14)** | **0.4778** | **0.8222** |

Both metrics are at or above the ungated arm — the gate cost nothing in aggregate on this eval set, on top of fixing the stability problem it was built for.

**q001 — confirmed fixed.** Now resolves via `dense_confident`, never touches rewrite. Deterministic by construction (dense search + exact-match have no sampling step) — this query cannot regress the way it did on Jul 13, regardless of what rewrite does to other queries.

**q004 — root cause found: not a rewrite problem, a gate-threshold tradeoff**. Widened-pool dense search on the raw query (no rewrite) puts T1690 at rank 1/15, distance 0.3336 — dense search alone already has the right answer. That distance is one of calibration's own 8 recorded "success" distances (same query, same computation), sitting in the 0.324–0.4464 zone where success and failure distances overlap — exactly the zone the conservative threshold (0.3146) was built to exclude from trust. q004 is the concrete, named cost of that choice: a query dense search already answers correctly, deferred to rewrite because its distance sits 0.019 above the trusted cutoff, where rewrite then answers it wrong. This supersedes the "why does rewrite consistently miss T1690" framing previously written here — rewrite's behavior on T1690 was never really the question; the gate's threshold placement is.
Not a new bug: the conservative threshold's design explicitly accepted this tradeoff (fewer trusted successes, in exchange for zero observed false positives), stated at calibration time. What changes is that the cost is no longer abstract — it's this one identified query, and by extension the ~4-query group of overlap-zone successes the threshold was built to sacrifice. Worth revisiting once the eval set grows past its current 13 gate-relevant queries — more data could either sharpen a cleaner threshold in this zone or justify the margin-based second signal already named, but deliberately not built, in `confidence_gate.py's` docstring.

**q015 — known limitation, not a new regression.** `dense_confident` on the `both` route, but only 0.5 recall (misses `T1190`). The `both` route's dense-search-first check runs an unfiltered whole-store search — it can look confident on one half of a two-part query without ever attempting the corpus split that might catch the other half. Same 0.5 recall q015 got
under rewrite before, so no regression — but it's the specific risk named in `confidence_gate.py`'s calibration docstring, now observed in practice rather than theoretical. Worth a per-route-type threshold check if the eval set grows and `both`-route traffic becomes a bigger fraction of it.

**q016 — the flagship Jul 13 instability example, now stable, and that's itself evidence.** Jul 13's Known Limitation 2 named q016 as its clearest case of rewrite non-determinism: three different outcomes across three runs, observed during active prompt patching. Across Jul 14's 3 frozen-prompt runs, q016 returns the identical result every time (recall 0.5, all 3 runs). The query that most demonstrated instability under active editing shows none under a frozen prompt — the strongest single data point for the prompt-editing-noise reframing above, stronger than the general "10 rewrite rows held stable" observation alone. What's exposed once the flakiness stops masking it is Limitation 1: CVE-2020-0796 sits at rank 5/15 in the retrieval pool — a dense-ranking miss, not an instability artifact. q016 and q015 share this root cause and belong in one follow-up issue, not two — Jul 13 already grouped them as "Known Limitation 1 (q015, q016)."

**Revised understanding of the Jul 13 "rewrite non-determinism" finding:** likely partially an artifact of concurrent prompt editing in that session, not purely a `temperature=0.0` sampling effect. Not proven — 3 samples on one frozen query set is suggestive, not conclusive — but worth stating as the current best explanation rather than repeating "rewrite is inherently unstable" unqualified going forward.

**mitre|medium delta (0.3333→0.3667 / 0.8000→1.0000), resolved**: two rows account for the full shift. q001 is the fix already documented above — Jul 13's confirmed run still had it failing (recall 0.0), Jul 14 resolves it via dense_confident. q006 is new: recall was already 1.0 under Jul 13's unconditional rewrite (precision 0.5, 2 chunks retrieved); Jul 14's gate lets it bypass rewrite (dense_confident, precision 0.3333, 3 chunks retrieved, one extra unrelated near-miss). Recall unaffected — the precision dip is a byproduct of the retrieval mechanism switching on a query that was correct either way, not a regression.

## Gated Faithfulness Re-Score - July 15

**Motivation.** The "Routed Faithfulness Results" above (mean 5.000, N=9 eligible / 6 ineligible) were measured against the routed arm at recall@3 = 0.5667. Gating raised recall@3 to 0.8222 (see Jul 14 results above), so several of the 6 previously-ineligible rows are almost certainly eligible now — the old N and mean no longer describe the current retrieval arm.

**Precondition, discovered this session.** No code path had ever connected gated retrieval to answer generation. `retrieve_for_route()` (the gate) was only ever called by the retrieval-only eval harness; `run_pipeline` — the function both the live app and `generation_capture.py` call — retrieved via plain `DB.semantic_search` regardless of gating. Fixed by adding a `use_confidence_gate` parameter to `run_pipeline`, valid only with `use_routing=True`: when set, the routed branch calls `retrieve_for_route` instead of `semantic_search`. Ungated paths (blind arm, and routed with the gate off) are untouched byte-for-byte — existing capture artifacts from those arms remain valid. A third capture arm (`generation_capture_gated.json`) was then run to produce answers actually generated against gated context, and faithfulness was re-scored against it, pointed at one of the three byte-identical `routed_with_fixes_gated` MLflow runs from Jul 14 via `recall_run_id` lineage.

### Results

| Arm | Faithfulness Mean | N Eligible | N Ineligible |
|---|---|---|---|
| Routed (Jul 14, stale) | 5.000 | 9 | 6 |
| **Routed + fixes + gate (Jul 15)** | **4.429** | **14** | **1** |

Reconciles: 14 + 1 = 15. The single gated-out row is **q004** — consistent with its Jul 14 root-cause diagnosis (dense search already correct at distance 0.3336, deferred to rewrite by the conservative threshold's design, rewrite answers it wrong). q004 being both the sole retrieval-ineligible row here and the concrete cost case named on Jul 14 is a second, independent confirmation of that diagnosis, not a new finding.

**The mean dropping from 5.000 to 4.429 is not a quality regression.** N grew from 9 to 14 because eligibility (recall@3 > 0) tracks the retrieval arm, and recall@3 jumped 0.5667 → 0.8222 under gating. A mean computed over a larger, more honestly-sampled set moving off a suspiciously perfect 5.000 is the expected result of judging five previously-unscored rows, not evidence that gating made answers less faithful. Comparing 5.000/N=9 to 4.429/N=14 directly, as if they measured the same thing, would repeat the exact cross-run metric mismatch this pipeline's reconciliation discipline exists to prevent.

### Judge reliability — bidirectional, not just lenient-on-refusal

Three rows carry the actual finding this run adds.

**q013 and q015 both scored 5 again** — the same refusal-trivially-passes pattern already documented on the routed arm (q013's blind→routed flip from 1→5 was shown there to be judge variance, not a caught error; q015 is the standing proof case for the judge's blind spot). Both recurring here, independently, on a different retrieval arm, is a second confirmation of an already-known pattern — not a new discovery on its own.

**q003 is new** — it was ineligible under the pre-gate arm (recall@3 = 0) and only entered the scored set because gating fixed its retrieval. The query asks *which specific built-in Windows utilities* decode or deobfuscate a payload. The retrieved context (T1140 + a payload-compression chunk) discusses the technique substantively but never names a utility — "built-in functionality of malware or... utilities present on the system," no names given. The model's refusal here is arguably *correct*: naming a specific utility would have been the actual fabrication. The judge scored it **1**, reasoning the context "provides substantial information" — true of the technique in general, not of the specific thing asked.

Set against q013 — whose context contains the literal answer twice ("Microsoft (Windows) Graphics Device Interface (GDI) Privilege Escalation Vulnerability," verbatim, across two chunks) and still refused, still scored 5 — the inconsistency is now bidirectional: the judge has under-penalized a wrongful refusal against a context with an exact match (q013, q015), and over-penalized a defensible refusal against a context with only generic relevance (q003), in the same eval set. There is no stable rule visible in the judge's behavior for how much context-relevance should excuse or convict a refusal.

**Revised statement of the judge limitation:** not simply "biased toward scoring refusals high." The judge's refusal-handling is inconsistent in both directions — it has no reliable relationship between how well the context actually answers the question and the score a refusal against that context receives. This is a stronger and more precise claim than the one-directional blind spot named after the Jul 13/14 runs, and it further supports deferring a correctness-against-gold metric rather than trying to patch the faithfulness rubric to catch refusals — the failure mode isn't "too lenient," it's "uncorrelated with the thing that should determine the score."

### Not addressed this run

- Whether q013/q015's refusals should be treated as retrieval-adjacent findings (context contains a partial or full answer the prompt's strict grounding rule still forces a refusal on) is the same over-refusal tradeoff already characterized in Baseline Failure Analysis — not re-litigated here.
- No change was made to the judge prompt or rubric in response to q003. The finding is recorded as evidence for the deferred correctness-vs-gold metric, not treated as something to patch via prompt engineering on the judge itself.