# RAG Threat Intelligence Assistant

A retrieval-augmented generation system for querying MITRE ATT&CK techniques and CISA KEV (Known Exploited Vulnerabilities) data, with agentic corpus routing and a full evaluation pipeline.

## What it does

Ask natural-language questions about attack techniques and known exploited vulnerabilities — e.g. "What technique does Log4Shell map to, and is it actively exploited?" — and get a grounded answer with source citations, instead of an LLM guessing from memory.

## Architecture

Query → Agentic Router (Gemini function-calling)
│
├── routes to MITRE ATT&CK collection
├── routes to CISA KEV collection
├── routes to both
└── skip retrieval (greetings, meta-queries)
│
▼
Retrieved context (ChromaDB, all-MiniLM-L6-v2 embeddings)
│
▼
Gemini 2.5 Flash generation → response + source citations

Routing exists because blind retrieval (always querying both corpora) wastes a retrieval pass on the wrong corpus for corpus-specific queries, and risks citing irrelevant chunks on queries that need no retrieval at all. Full reasoning for this design choice — including why it's kept despite a flat precision delta in evaluation — is in [`docs/adr/001-agentic-routing.md`](./docs/adr/001-agentic-routing.md).

## Tech stack

- **Retrieval:** ChromaDB, `all-MiniLM-L6-v2` embeddings
- **Generation & routing:** Gemini 2.5 Flash via Vertex AI (`google.genai` SDK)
- **Corpora:** MITRE ATT&CK (STIX bundle), CISA KEV (JSON feed)
- **Experiment tracking:** MLflow (local server, SQLite backend)
- **CI:** GitHub Actions — ingestion + retrieval smoke tests (generation excluded from CI: costs money per run, flaky, needs live secrets a smoke test shouldn't require)

## Evaluation pipeline

This project is evaluated on retrieval quality and generation faithfulness, not just "does it run."

**Eval set:** 20 hand-written queries spanning MITRE ATT&CK and CISA KEV, with manually authored gold answers (technique IDs / CVEs) — see `eval_set.json`.

**Metrics:**
- **Retrieval:** Precision@3, Recall@3 against expected technique IDs / CVEs
- **Faithfulness:** LLM-as-judge (Gemini 2.5 Flash), 1–5 scale, scoring whether generated claims actually appear in retrieved context

**Baseline (blind retrieval, always queries both corpora):**
- Precision@3 = 0.2444, Recall@3 = 0.5667
- Faithfulness mean = 4.444

**Routed (agentic routing selects corpus before retrieval):**
- Precision@3 = 0.2444, Recall@3 = 0.5667 (delta 0.0000, 0 misroutes)
- Faithfulness mean = 5.000 (read with a caveat — see `eval_pipeline.md`; partly driven by refusal rows scoring 5 for making no claims, not a clean win)

Routing was further validated with a follow-up stress test — 8 deliberately adversarial cross-corpus queries with no explicit IDs — which held at 0 misroutes on all unambiguous cases.

Full methodology, eval-set construction, failure-mode analysis, and the stress-test breakdown: [`eval_pipeline.md`](./eval_pipeline.md).

**Known limitation:** the residual precision ceiling traces to the retrieval
layer, not routing — exact-identifier lookup failures and vocabulary-mismatch queries. A hybrid-retrieval fix is scoped but not yet built (tracked in the ADR as a follow-up).

## Setup

```bash
git clone https://github.com/sayantika-mlsec/RAG-Powered-Threat-Intelligence-Assistant.git
cd RAG-Powered_Threat-Intelligence-Assistant
pip install -r requirements.txt
```

Requires a GCP project with Vertex AI enabled and application default credentials configured (`gcloud auth application-default login`).

## Usage

```bash
python app.py
```

## Project structure

.
├── app.py                    # main query interface + routing
├── gemini_client.py          # shared Vertex AI client
├── generation_capture.py     # eval generation runs
├── eval_retrieval.py         # retrieval metrics
├── eval_faithfulness.py      # faithfulness scoring
├── inspect_routes.py         # routing inspection utility
├── stress_test_routes.py     # ad-hoc routing stress test
├── eval_set.json             # 20 hand-written eval queries
├── eval_pipeline.md          # full eval methodology + results
├── docs/
│   └── adr/
│       └── 001-agentic-routing.md
└── requirements.txt