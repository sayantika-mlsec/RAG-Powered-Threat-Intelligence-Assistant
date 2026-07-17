"""Standalone routing inspection — NOT a committed test.

Runs route_query() over the eval set and prints predicted vs expected route per
row so misroutes are visible alongside the model's own reasoning. This is the
manual "do the routes look sane" gate before wiring routing into retrieval; it
asserts nothing and is safe to re-run (idempotent against a fixed eval_set.json).

A per-call throttle keeps the run under the per-minute request quota: routing
calls share the project's quota pool, so spacing them out lets all rows
complete in one pass instead of erroring partway through. Default is 6s/call
(~10 req/min) — conservative for Vertex's paid tier, but left as-is until the
actual ceiling is confirmed in the Cloud Console. Raise --delay if rate-limited,
lower it once you've confirmed headroom.

Usage:
    python inspect_routes.py                  # all rows, throttled
    python inspect_routes.py --ids q014 q019  # just the watch rows
    python inspect_routes.py --sample 10      # first N rows
    python inspect_routes.py --delay 8        # raise throttle if still rate-limited
"""

import argparse
import json
import time
from pathlib import Path

from dotenv import load_dotenv

# Load .env BEFORE importing config, unconditionally. config.py reads
# GCP_PROJECT_ID at import time — if this file (or anything it imports) reaches
# `import config` before .env is loaded, that import fails outright.
load_dotenv(dotenv_path=Path(__file__).parent / ".env", override=True)

from gemini_client import get_client
from routing import route_query

import config

THROTTLE_SECONDS = 6  # ~10 req/min — see module docstring


def load_rows():
    with open(config.EVAL_SET_PATH) as f:
        return json.load(f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ids", nargs="*", help="only these query ids")
    ap.add_argument("--sample", type=int, help="first N rows")
    ap.add_argument(
        "--delay",
        type=float,
        default=THROTTLE_SECONDS,
        help="seconds to wait between calls (raise if rate-limited, lower if quota allows)",
    )
    args = ap.parse_args()

    client = get_client()

    rows = load_rows()
    if args.ids:
        rows = [r for r in rows if r["id"] in args.ids]
    elif args.sample:
        rows = rows[: args.sample]

    mismatches = []
    errors = []
    print(f"throttle: {args.delay}s between calls | {len(rows)} rows\n")
    # tier has no expected_tier ground truth yet (eval_set.json isn't tagged
    # for it) — printed for visual sanity-check only, not scored. Tier
    # accuracy gets a real eval, once tagged runs exist.
    print(f"{'id':<6} {'expected':<11} {'predicted':<11} {'tier':<6} {'ok':<3} query")
    print("-" * 100)

    for i, r in enumerate(rows):
        expected = r["expected_route"]
        tier = "?"
        try:
            decision = route_query(r["query"], client)
            predicted = decision.route.value
            tier = decision.tier.value
            reasoning = decision.reasoning
        except Exception as e:
            predicted = f"ERROR:{type(e).__name__}"
            reasoning = str(e)
            errors.append((r["id"], reasoning))

        ok = "✓" if predicted == expected else "✗"
        if predicted != expected and not predicted.startswith("ERROR:"):
            mismatches.append((r["id"], expected, predicted, r["query"], reasoning))

        q = r["query"][:60] + ("…" if len(r["query"]) > 60 else "")
        print(f"{r['id']:<6} {expected:<11} {predicted:<11} {tier:<6} {ok:<3} {q}")

        # Throttle between calls, but not after the last one.
        if i < len(rows) - 1:
            time.sleep(args.delay)

    print("-" * 100)
    matched = sum(
        1
        for r in rows
        if not any(r["id"] == m[0] for m in mismatches)
        and not any(r["id"] == e[0] for e in errors)
    )
    print(f"{matched}/{len(rows)} match expected route")

    if mismatches:
        print("\nMISMATCHES (inspect reasoning — misroute, or is the label wrong?):")
        for qid, exp, pred, query, reasoning in mismatches:
            print(f"\n  {qid}: expected={exp} predicted={pred}")
            print(f"    query: {query}")
            print(f"    model reasoning: {reasoning}")

    if errors:
        print(f"\nERRORS ({len(errors)} rows — likely rate limit if 429/RESOURCE_EXHAUSTED):")
        for qid, msg in errors:
            print(f"  {qid}: {msg[:120]}")
        print("\nIf rate-limited, re-run with a larger --delay (e.g. --delay 8).")


if __name__ == "__main__":
    main()