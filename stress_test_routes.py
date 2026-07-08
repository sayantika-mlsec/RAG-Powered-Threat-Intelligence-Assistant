"""
Routing stress test — adversarial and edge-case queries against route_query().

Logs routing decisions only (no generation, no judge call — zero Gemini
generation quota spent, just the router's function-calling pass).

Expected values must match the Route enum's .value strings exactly:
"mitre_only", "kev_only", "both", "skip".

q024 is EXPECTED to misroute — it reproduces the known vocabulary-mismatch
failure at the routing layer (plain-English phrasing carries no KEV signal).
q025 is ambiguous by design: both "skip" and "both" are accepted (pipe-separated).
"""
from dotenv import load_dotenv
load_dotenv()

from gemini_client import get_client
from app import route_query

STRESS_QUERIES = [
    ("q021", "What technique does Log4Shell map to, and is it actively exploited?", "both"),
    ("q022", "Which actively exploited vulnerabilities involve lateral movement?", "both"),
    ("q023", "Is the MOVEit vulnerability linked to a known credential access technique?", "both"),
    ("q024", "How do attackers get in through unpatched internet-facing software?", "kev_only"),  # known vocab-mismatch case
    ("q025", "Tell me about ransomware.", "skip|both"),  # ambiguous by design — either route accepted
    ("q026", "T1190", "mitre_only"),
    ("q027", "CVE-2021-34473", "kev_only"),
    ("q028", "What did we just talk about?", "skip"),
]

client = get_client()


def run_stress_test():
    rows = []
    for qid, query, expected in STRESS_QUERIES:
        try:
            result = route_query(query, client)
            actual = result.route.value  # plain string: "both", "kev_only", etc.
        except Exception as e:
            actual = f"ERROR: {e}"
        accepted = expected.split("|")  # supports multi-accept cases like q025
        misroute = "no" if actual in accepted else "yes"
        rows.append((qid, query, expected, actual, misroute))
    return rows


def print_markdown_table(rows):
    print("| Query ID | Query | Expected Route | Actual Route | Misroute? |")
    print("|---|---|---|---|---|")
    for qid, query, expected, actual, misroute in rows:
        # escape pipe chars in query text just in case
        safe_query = query.replace("|", "\\|")
        print(f"| {qid} | {safe_query} | {expected} | {actual} | {misroute} |")


if __name__ == "__main__":
    results = run_stress_test()
    print_markdown_table(results)