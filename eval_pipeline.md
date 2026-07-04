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

Gold answers are written **by hand, not AI-generated.** This is the load-bearing decision of the whole pipeline: an AI-generated gold answer is just another model output, so scoring against it measures agreement between two models, not correctness Hand-written gold answers — verified against the actually-ingested chunks — are the only real ground truth available. (As the failure analysis shows, even hand-written golds can encode assumptions the system's prompt forbids — see q009 — which is itself a finding, not a flaw in the approach.)

The 20 rows split into a 15-row Group A (the reconciliation total used throughout) plus rows reserved for later expansion. Thin technique categories and new queries (`q021+`) are grown alongside the routing work.

## Retrieval metrics — precision@K and recall@K

**Precision@K** asks: of the K chunks retrieved, how many were relevant?
**Recall@K** asks: of all the relevant chunks that exist, how many did we get in the top K? Both are measured at **K=3** against `expected_technique_ids`.

Recall is the metric that gates the rest of the pipeline. A row with recall@3 = 0 — the right chunk never came back - is**ineligible for faithfulness scoring**, because faithfulness measures answer-vs-retrieved-context, and if retrieval failed there is no correct context to be faithful to. Scoring those rows would measure the generator's behaviour on garbage input, which is a different question. They are gated out, counted, and analysed separately (see failure analysis).

Per-row recall is emitted as an MLflow artifact with fail-loud invariants: a K-pin guard (the K used must match the K claimed), a corpus stamp (chunk count recorded so a silently-changed corpus is detectable), and a reconciliation check. The eligibility decision for every downstream stage is *read from this artifact by run id*, never recomputed — so faithfulness is provably tied to the exact retrieval run that produced eligibility.

## Faithfulness metric — LLM-as-judge

Faithfulness asks one question: do the generated answer's claims appear in the retrieved context? It is scored by **Gemini 2.5 Flash as judge**, on a 1–5 rubric (5 = every claim traces to context; 1 = central claim contradicts context or is fabricated). The judge sees only the answer and the context — **gold answers are not an input.** Faithfulness is grounding, not correctness; the two are separated on purpose.

**Why LLM-as-judge over cosine similarity.** A cheaper alternative is cosine similarity between answer and context embeddings. It was rejected as the default (documented as a fallback if cost/latency forces it) because similarity measures *topical overlap*, not *entailment* — an answer can be highly similar to context it actually contradicts. A rubric-driven judge reasons about whether each claim is supported, which is the property faithfulness is supposed to capture.

**Judge implementation note.** The judge runs on the new `google.genai` SDK specifically to disable model thinking (`ThinkingConfig(thinking_budget=0)`): `gemini-2.5-flash` thinks by default, and the thinking tokens consumed the output budget and truncated the JSON reply. The old `google.generativeai` SDK has no `ThinkingConfig`, which forced the judge — and only the judge — onto the new SDK. Generation and ingest remain on the old SDK; the two coexist intentionally and transitionally. (This two-SDK decision is an ADR candidate.)

**Known limitation.** The judge is structurally blind to wrongful refusal — an answer that makes no claims trivially passes the grounding check. This is detailed in the failure analysis with q015 as the proof case, and is the concrete reason a separate correctness-against-gold metric is necessary (deferred, but justified by data).

## Logging contract

The faithfulness score is **never logged bare.** It travels with N (eligible row count), the gated-out count, and `recall_run_id` lineage, in a single MLflow run. Reconciliation (`N_eligible + gated_out == 15`) runs *before* any scoring — if the artifacts disagree about the eval set, the run stops rather than scoring against an inconsistent denominator. The empty-eligible case (N=0) logs null rather than dividing by zero. Bad judge output raises rather than coercing to a middle value.

## Baseline numbers (single-pass RAG)

| Metric | Value |
|---|---|
| Precision@3 | 0.2444 |
| Recall@3 | 0.5667 |
| Faithfulness mean | 4.444 (N=9 eligible) |
| Gated out (recall@3 = 0) | 6 |
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

This is not a judge bug to fix. It is an inherent property of grounding-only scoring: **faithfulness penalizes hallucination but is blind to under-answering. A model that refuses everything scores near-perfect faithfulness.** This is the concrete rows-behind-it evidence that faithfulness ≠ correctness, and the reason a separate correctness-against-gold metric is necessary (deferred, but now justified by data rather than assertion).

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
- **Correct corpus selection did not move top-3.** The store imbalance the routing layer was built to counter (540 MITRE chunks vs 1600 KEV) does distort deeper ranks — a MITRE query under `both` competes against 1600 KEV chunks — but for this eval set the correct chunks were *already ranking in the top 3 under blind retrieval*. Filtering out wrong-corpus chunks that sat at rank 4+ changed nothing about which three chunks won the top-3 slots. Routing removed noise that K=3 never saw.

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

Agentic routing held precision constant at K=3 while eliminating misroutes; the residual precision ceiling is a retrieval-layer problem, not a corpus-selection one. This is the measured evidence — not a prediction — behind the hybrid-retrieval ADR: routing is the right tool for corpus selection and the wrong tool for exact-identifier and vocab-mismatch lookup, and the data now shows exactly that separation. A flat precision delta with a clean misroute count is a stronger localization of the next bottleneck than a precision bump would have been.

> **Faithfulness (routed arm): pending.** The routed retrieval run above is only the precision half of the A/B. Routed faithfulness has not yet been scored, and precision-flat says nothing about whether routing changed answer grounding — that is a separate distribution. Before scoring, `RECALL_BASELINE_RUN_ID` must be repointed at *this* routed retrieval run's id, or routed answers get gated against blind eligibility. Routed faithfulness numbers and any routed row in the baseline table are deliberately omitted until that run lands.