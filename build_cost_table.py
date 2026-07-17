"""Cost/latency-per-tier table

Reads a tier-tagged generation_capture artifact and computes real cost
(current Gemini pricing, pulled 2026-07-17 from
https://ai.google.dev/gemini-api/docs/pricing) and latency, aggregated by
tier. Only rows with generation_ok=True and mode != 'no_retrieval' are
included — the skip path never calls ThreatAnalyzer at all, so it has no
real usage data to aggregate. That's a "no data exists" fact, not a
judgment about the flash tier those rows happen to carry.

Also computes what the SAME set of calls would have cost at a uniform
tier (all-Flash, all-Pro) — this is what actually answers whether tiering
is worth it. A percentage split isn't a dollar figure.

Usage:
    python build_cost_table.py                                  # default path
    python build_cost_table.py --input generation_capture_gated.json
"""

import argparse
import json
from pathlib import Path

# Pricing pulled 2026-07-17 from https://ai.google.dev/gemini-api/docs/pricing
# (Developer API / AI Studio Standard tier, prompts <=200k tokens — nothing
# in this project's prompts is remotely close to that ceiling).
#
# NOT CONFIRMED identical to Vertex AI pricing, which is what this project
# actually bills against — Google has historically kept Vertex and the
# Developer API aligned per-token, but that hasn't been checked against the
# real GCP billing console. Treat these dollar figures as directionally
# correct, not final, until verified there.
PRICING_PER_1M_USD = {
    "flash": {"input": 0.30, "output": 2.50},
    "pro":   {"input": 1.25, "output": 10.00},
}


def load_rows(path: str) -> list[dict]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return data["rows"]


def usable_rows(rows: list[dict]) -> list[dict]:
    """
    Rows with real usage data to cost out: generation actually happened
    (generation_ok=True) and it wasn't the no-retrieval skip path. Skip
    never reaches ThreatAnalyzer (see app.py's _no_retrieval_response), so
    latency/token fields are None there regardless of tier — excluding
    these rows is a data-availability fact, not a tiering judgment.
    """
    return [
        r for r in rows
        if r.get("generation_ok") and r.get("mode") != "no_retrieval"
    ]


def row_cost_usd(row: dict, tier: str) -> float:
    """
    Cost for one row, priced at `tier` — a parameter, not read from
    row['tier']. Same formula prices a row at its ACTUAL tier, or at a
    HYPOTHETICAL tier for the all-flash/all-pro comparison below.

    output + thinking tokens are billed together as "output" per Google's
    pricing page ("Output price (including thinking tokens)"). Thinking
    was deliberately left on for these calls (see eval_pipeline.md's Jul
    25 entry) — it's real, unavoidable spend on this data, not overhead to
    exclude from the number.
    """
    prices = PRICING_PER_1M_USD[tier]
    input_tokens = row["input_tokens"] or 0
    output_tokens = (row["output_tokens"] or 0) + (row["thinking_tokens"] or 0)
    return (input_tokens * prices["input"] + output_tokens * prices["output"]) / 1_000_000


def aggregate_by_tier(rows: list[dict]) -> dict:
    """Real cost/latency, grouped by each row's ACTUAL dispatched tier."""
    by_tier: dict[str, list[dict]] = {"flash": [], "pro": []}
    for r in rows:
        tier = r.get("tier")
        if tier not in by_tier:
            raise ValueError(
                f"Row {r.get('id')} has unexpected tier {tier!r} — expected "
                f"'flash' or 'pro'. Refusing to silently drop it from the table."
            )
        by_tier[tier].append(r)

    summary = {}
    for tier, tier_rows in by_tier.items():
        if not tier_rows:
            summary[tier] = {"n": 0}
            continue
        costs = [row_cost_usd(r, tier) for r in tier_rows]
        latencies = [r["latency_seconds"] for r in tier_rows]
        input_toks = [r["input_tokens"] or 0 for r in tier_rows]
        output_toks = [r["output_tokens"] or 0 for r in tier_rows]
        thinking_toks = [r["thinking_tokens"] or 0 for r in tier_rows]
        summary[tier] = {
            "n": len(tier_rows),
            "total_cost_usd": sum(costs),
            "avg_cost_usd": sum(costs) / len(costs),
            "avg_latency_s": sum(latencies) / len(latencies),
            "avg_input_tokens": sum(input_toks) / len(input_toks),
            "avg_output_tokens": sum(output_toks) / len(output_toks),
            "avg_thinking_tokens": sum(thinking_toks) / len(thinking_toks),
        }
    return summary


def hypothetical_totals(rows: list[dict]) -> dict:
    """
    What the SAME set of calls would have cost at a uniform tier, holding
    each row's actual token counts fixed. An approximation, not a
    re-measurement: a real all-Pro run might produce different token
    counts than Flash did on the same query (Pro may reason longer or
    shorter). This shows the SHAPE of the tradeoff; a precise number would
    need an actual all-Pro / all-Flash capture run, not this estimate.
    """
    return {
        "actual_tiered_usd": sum(row_cost_usd(r, r["tier"]) for r in rows),
        "all_flash_usd": sum(row_cost_usd(r, "flash") for r in rows),
        "all_pro_usd": sum(row_cost_usd(r, "pro") for r in rows),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="generation_capture_routed.json")
    args = ap.parse_args()

    rows = load_rows(args.input)
    rows = usable_rows(rows)

    if not rows:
        raise RuntimeError(
            f"No usable rows (generation_ok=True, mode != 'no_retrieval') "
            f"found in {args.input}. Nothing to report."
        )

    by_tier = aggregate_by_tier(rows)
    hypo = hypothetical_totals(rows)

    print(f"Cost/latency table — {args.input}")
    print(f"({len(rows)} usable rows: generation_ok=True, excludes skip)\n")

    print(f"{'Tier':<8} {'N':<4} {'Avg latency':<13} {'Avg cost':<12} {'Total cost':<13} {'Avg in':<8} {'Avg out':<8} {'Avg thinking'}")
    print("-" * 92)
    for tier in ("flash", "pro"):
        s = by_tier[tier]
        if s["n"] == 0:
            print(f"{tier:<8} 0    (no rows)")
            continue
        print(
            f"{tier:<8} {s['n']:<4} {s['avg_latency_s']:<13.2f} "
            f"${s['avg_cost_usd']:<11.5f} ${s['total_cost_usd']:<12.5f} "
            f"{s['avg_input_tokens']:<8.0f} {s['avg_output_tokens']:<8.0f} {s['avg_thinking_tokens']:.0f}"
        )

    print(f"\n--- Hypothetical: same {len(rows)} calls at a uniform tier ---")
    print(f"Actual (tiered):  ${hypo['actual_tiered_usd']:.5f}")
    print(f"All-Flash:        ${hypo['all_flash_usd']:.5f}")
    print(f"All-Pro:          ${hypo['all_pro_usd']:.5f}")

    savings_vs_all_pro = hypo["all_pro_usd"] - hypo["actual_tiered_usd"]
    premium_vs_all_flash = hypo["actual_tiered_usd"] - hypo["all_flash_usd"]
    pro_n = by_tier["pro"]["n"]
    print(
        f"\nTiering saves ${savings_vs_all_pro:.5f} vs. all-Pro "
        f"({100 * savings_vs_all_pro / hypo['all_pro_usd']:.1f}% cheaper)."
    )
    print(
        f"Tiering costs ${premium_vs_all_flash:.5f} more than all-Flash "
        f"({100 * premium_vs_all_flash / hypo['all_flash_usd']:.1f}% pricier) — "
        f"the premium for routing {pro_n} of {len(rows)} queries to better reasoning."
    )


if __name__ == "__main__":
    main()