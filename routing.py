"""Agentic routing layer for the RAG threat-intel assistant.

The router makes two independent decisions per query before any retrieval or
generation runs:
  1. Which collection(s) to query, or whether to skip retrieval (Route).
  2. How much reasoning the query needs, which determines the generation
     model (ModelTier) — a single-chunk lookup vs. a cross-chunk/cross-corpus
     synthesis question have different failure profiles, and dispatching
     both to the same model either overpays on easy queries or under-
     provisions hard ones.

It does NOT decompose multi-clause queries or fix embedding misalignment;
those remain separate concerns (query decomposition, reranking).

Built on the `google.genai` SDK (consistent with eval_faithfulness.py). Both
decisions are produced via structured output (response_schema) in a single
call, not keyword if/else, so they're made on query meaning rather than
string matching.
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


class ModelTier(str, Enum):
    """Which model generation should use, based on how much reasoning the
    query needs — independent of which corpus/corpora it's routed to. A
    BOTH-routed query is often (not always) PRO; a single-corpus query is
    usually FLASH unless it still requires combining multiple facts within
    that one corpus."""

    FLASH = "flash"  # single-chunk lookup: retrieve, restate/extract — no
                      # cross-fact inference required
    PRO = "pro"       # multi-hop synthesis: answer requires combining facts
                      # from separate chunks (same corpus or across corpora)


class RoutingDecision(BaseModel):
    """A single routing decision covering both corpus selection and model
    tier.

    `reasoning` is required, not optional: it costs almost nothing to generate
    and is the only thing available to inspect when a route or tier looks
    wrong during stress-testing.
    """

    route: Route
    tier: ModelTier
    reasoning: str


# Model identifier kept as a module constant so it's logged/changed in one place.
_ROUTER_MODEL = "gemini-2.5-flash"

# Few-shot examples are drawn from the real eval set (q001, q011, q015, q017) so
# the router is anchored on the actual query distribution it will see, not
# invented phrasings. The `both` anchor (q015) is the category most prone to
# misfire on route, and it doubles as the PRO tier anchor here since it's a
# genuine two-fact synthesis case — the model has to connect a specific CVE to
# a technique, not just retrieve and restate one chunk.
_ROUTING_PROMPT = """You are a retrieval router for a threat-intelligence assistant with two corpora:
- MITRE ATT&CK: adversary techniques, tactics, and procedures (TTPs), identified by technique IDs like T1059.
- CISA KEV: specific CVEs known to be exploited in the wild, identified by CVE IDs.

Classify the query on TWO independent axes:

ROUTE — which corpus/corpora to search:
- mitre_only: about techniques/tactics/TTPs, with no specific CVE.
- kev_only: about a specific CVE, a named product's exploited vulnerability, or whether something is in the exploited catalog — with no technique-mapping ask. This includes questions about products that may NOT be in the catalog: still route here so the corpus can be searched and correctly report no match.
- both: genuinely spans techniques AND a specific CVE / exploited vulnerability.
- skip: greeting, off-topic, or meta-question — nothing either corpus can answer, so no retrieval.

TIER — how much reasoning the answer needs:
- flash: the answer lives in a single chunk. Retrieve and restate/extract — no inference connecting separate facts.
- pro: the answer requires combining facts from more than one chunk — for example, linking a specific CVE to the technique it enables, or connecting two separate techniques. This is NOT the same as route=both: a single-corpus query can still need pro if it requires combining multiple chunks within that one corpus.

Examples:
[query]: What technique covers hiding command-and-control traffic by disguising it as legitimate protocols or padding it with junk data?
[route]: mitre_only
[tier]: flash

[query]: What is CVE-2012-4681 and why is it on CISA's known-exploited list?
[route]: kev_only
[tier]: flash

[query]: Is the remote code execution flaw in Cisco IP Phones' web server listed as actively exploited, and what attack technique covers exploiting an internet-facing service like this?
[route]: both
[tier]: pro

[query]: What's the capital of Italy?
[route]: skip
[tier]: flash

Classify this query:
[query]: {query}
"""


def build_routing_prompt(query: str) -> str:
    """Return the routing prompt with the query interpolated."""
    return _ROUTING_PROMPT.format(query=query)


def route_query(query: str, client: genai.Client) -> RoutingDecision:
    """Classify a query into a single Route + ModelTier before retrieval.

    Fail-loud on both failure paths:
      - The enums (via the response_schema) reject an out-of-set route or
        tier value.
      - An explicit guard rejects an unparseable response (resp.parsed is
        None), which the enums cannot catch because there is no value to
        validate.
    A silent default is deliberately NOT used for either field: it would
    reintroduce the blind-retrieval / one-size-fits-all-model behavior this
    layer exists to remove.
    """
    resp = client.models.generate_content(
        model=_ROUTER_MODEL,
        contents=build_routing_prompt(query),
        config=types.GenerateContentConfig(
            temperature=0.0,
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

    # tier is meaningless on skip — nothing downstream ever generates against
    # it. Enforced in code rather than left to the model's few-shot behavior,
    # so a skip row can never surface a spurious tier mismatch later.
    if decision.route == Route.SKIP:
        decision.tier = ModelTier.FLASH

    return decision