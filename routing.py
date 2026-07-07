"""Agentic routing layer for the RAG threat-intel assistant.

The router classifies each query into exactly one Route *before* any retrieval
runs. It decides which collection(s) to query, or whether to skip retrieval —
nothing else. It does NOT decompose multi-clause queries or fix embedding
misalignment; those are separate concerns (query decomposition, reranking).

Built on the `google.genai` SDK (consistent with eval_faithfulness.py). The route
is produced via structured output (response_schema), not keyword if/else, so the
decision is made on query meaning rather than string matching.
"""

from enum import Enum

from google import genai
from google.genai import types
from pydantic import BaseModel


class Route(str, Enum):
    """The four retrieval routes. Inherits from str so the value serializes as a
    plain string and compares equal to its string form (lets a decision compare
    directly against eval_set.json's expected_route)."""

    MITRE_ONLY = "mitre_only"   # technique/tactic/TTP queries, no specific CVE signal
    KEV_ONLY = "kev_only"       # specific CVE / exploited-in-the-wild; also the restraint case
    BOTH = "both"               # query genuinely spans both corpora
    SKIP = "skip"               # greetings, off-topic, meta-questions; no search


class RoutingDecision(BaseModel):
    """A single routing decision.

    `reasoning` is required, not optional: it costs almost nothing to generate and
    is the only thing available to inspect when a route looks wrong during
    stress-testing.
    """

    route: Route
    reasoning: str


# Model identifier kept as a module constant so it's logged/changed in one place.
_ROUTER_MODEL = "gemini-2.5-flash"

# Few-shot examples are drawn from the real eval set (q001, q011, q015, q017) so
# the router is anchored on the actual query distribution it will see, not
# invented phrasings. The `both` anchor (q015) is the category most prone to
# misfire, so it uses a genuine cross-corpus query.
_ROUTING_PROMPT = """You are a retrieval router for a threat-intelligence assistant with two corpora:
- MITRE ATT&CK: adversary techniques, tactics, and procedures (TTPs), identified by technique IDs like T1059.
- CISA KEV: specific CVEs known to be exploited in the wild, identified by CVE IDs.

Classify the query into exactly one route:
- mitre_only: about techniques/tactics/TTPs, with no specific CVE.
- kev_only: about a specific CVE, a named product's exploited vulnerability, or whether something is in the exploited catalog — with no technique-mapping ask. This includes questions about products that may NOT be in the catalog: still route here so the corpus can be searched and correctly report no match.
- both: genuinely spans techniques AND a specific CVE / exploited vulnerability.
- skip: greeting, off-topic, or meta-question — nothing either corpus can answer, so no retrieval.

Examples:
[query]: What technique covers hiding command-and-control traffic by disguising it as legitimate protocols or padding it with junk data?
[route]: mitre_only

[query]: What is CVE-2012-4681 and why is it on CISA's known-exploited list?
[route]: kev_only

[query]: Is the remote code execution flaw in Cisco IP Phones' web server listed as actively exploited, and what attack technique covers exploiting an internet-facing service like this?
[route]: both

[query]: What's the capital of Italy?
[route]: skip

Classify this query:
[query]: {query}
"""


def build_routing_prompt(query: str) -> str:
    """Return the routing prompt with the query interpolated."""
    return _ROUTING_PROMPT.format(query=query)


def route_query(query: str, client: genai.Client) -> RoutingDecision:
    """Classify a query into a single Route before retrieval.

    Fail-loud on both failure paths:
      - The enum (via the response_schema) rejects an out-of-set route value.
      - An explicit guard rejects an unparseable response (resp.parsed is None),
        which the enum cannot catch because there is no value to validate.
    A silent default to `both` is deliberately NOT used: it would reintroduce the
    blind-retrieval behavior this layer exists to remove.
    """
    resp = client.models.generate_content(
        model=_ROUTER_MODEL,
        contents=build_routing_prompt(query),
        config=types.GenerateContentConfig(
            thinking_config=types.ThinkingConfig(thinking_budget=0),
            response_mime_type="application/json",
            response_schema=RoutingDecision,
        ),
    )

    decision = resp.parsed
    if decision is None:
        raise RuntimeError(
            f"Router returned an unparseable response for query: {query!r}"
        )
    return decision