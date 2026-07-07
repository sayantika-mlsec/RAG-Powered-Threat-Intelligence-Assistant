"""
Logs routing decisions only (no generation, no judge call — zero Gemini
generation quota spent, just the router's function-calling pass).
"""
from dotenv import load_dotenv
load_dotenv()

from gemini_client import get_client  
from app import route_query 

STRESS_QUERIES = [
    ("q021", "What technique does Log4Shell map to, and is it actively exploited?", "both"),
    ("q022", "Which actively exploited vulnerabilities involve lateral movement?", "both"),
    ("q023", "Is the MOVEit vulnerability linked to a known credential access technique?", "both"),
    ("q024", "How do attackers get in through unpatched internet-facing software?", "kev"), 
    ("q025", "Tell me about ransomware.", "both"),  # ambiguous by design — no single "correct" answer
    ("q026", "T1190", "mitre"),
    ("q027", "CVE-2021-34473", "kev"),
    ("q028", "What did we just talk about?", "skip"),
]

client = get_client()

def run_stress_test():
    rows = []
    for qid, query, expected in STRESS_QUERIES:
        try:
            actual = route_query(query, client)  
        except Exception as e:
            actual = f"ERROR: {e}"
        misroute = "yes" if actual != expected else "no"
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