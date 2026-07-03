# Produces the artifact the faithfulness step (Issue 11) consumes: for every eval
# row, the generated answer plus the EXACT context the LLM saw (post-truncation,
# via result.context_used), keyed on `id`. Runs once per arm; faithfulness reads
# this and makes zero retrieval or generation calls — guaranteeing the judged
# context matches the context behind each answer.
#
# A/B CHANGE: capture now drives app.run_pipeline instead of calling
# semantic_search + generate_answer directly. use_routing is the ONLY variable
# between the two arms, so both arms are byte-identical except the flag — the
# single-variable discipline the whole routing branch exists to preserve.
#
#     use_routing=False -> blind baseline (whole store) -> generation_capture.json
#     use_routing=True  -> agentic (route decides corpus) -> generation_capture_routed.json
#
# Scope: capture records ALL rows, including ineligible ones AND skip-routed ones.
# The eligibility gate (recall@3 > 0) and the skip gate both live DOWNSTREAM
# (retrieval + faithfulness steps), not here — but this file records `route` and
# `mode` on every row so those steps can gate correctly. Rows whose generation
# fails are recorded WITH A FLAG, never skipped, so every eval row maps to
# exactly one record.

import argparse
import json
import logging
import time
from pathlib import Path

# run_pipeline is the single shared entry point (same function the Gradio UI
# calls). Importing it triggers app.py's module-load init of DB / ANALYZER /
# ROUTER_CLIENT, so we do NOT instantiate our own here — capture and the live
# app run through byte-identical retrieval + generation + routing code.
# run_pipeline now accepts n_results; capture pins it to config.RETRIEVAL_TOP_K
# (k=3) so BOTH arms match the pinned baseline capture (which used k=3), not the
# live UI default of N_RESULTS=5. Requires the one-line app.py signature change:
#   def run_pipeline(query, *, use_routing, n_results=N_RESULTS): ...
from app import run_pipeline
import config

# Seconds to sleep between live LLM calls, to stay under the free-tier
# per-minute rate limit. Free tier is ~10 RPM, so ~6s spacing keeps us safe.
# Applied only between rows that actually hit the API — reused rows cost nothing.
THROTTLE_SECONDS = 6

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
)
logger = logging.getLogger(__name__)


def capture_row(query: str, row_id: str, *, use_routing: bool) -> dict:
    """
    Retrieve + generate for one eval row via run_pipeline. Records the answer and
    the post-truncation context the LLM actually saw (result.context_used), plus
    the route taken and the result mode — both needed for downstream gating.

    Three outcomes, all recorded (never dropped):
      - retrieved success : generation_ok=True, mode='retrieved', real context
      - skip success      : generation_ok=True, mode='no_retrieval', EMPTY context
                            (route='skip' — gated out of retrieval AND faithfulness
                            scoring downstream; a no-claims answer trivially passes
                            grounding, so it must not enter the faithfulness set)
      - failure           : generation_ok=False, answer null, flagged
    """
    result, _search_results, route_value = run_pipeline(
        query, use_routing=use_routing, n_results=config.RETRIEVAL_TOP_K
    )

    if result.success:
        # Skip path: no retrieval happened, so there is no grounding context.
        # Force empty context regardless of what the result object carries, so a
        # skip row can never be mistaken for a grounded answer downstream.
        is_skip = getattr(result, "mode", "retrieved") == "no_retrieval"
        retrieved_context = [] if is_skip else result.context_used
        return {
            "id": row_id,
            "query": query,
            "generation_ok": True,
            "mode": getattr(result, "mode", "retrieved"),
            "route": route_value,                         # None on the blind arm
            "generated_answer": result.answer,
            "retrieved_context": retrieved_context,       # post-truncation / empty on skip
            "citations": result.source_citations,
            "error": None,
        }

    logger.warning(f"Generation failed for {row_id}: {result.error}")
    return {
        "id": row_id,
        "query": query,
        "generation_ok": False,
        "mode": getattr(result, "mode", "retrieved"),
        "route": route_value,
        "generated_answer": None,
        "retrieved_context": [],
        "citations": [],
        "error": result.error,
    }


def _load_existing(out_path: str) -> dict[str, dict]:
    """
    Load already-captured rows from a prior run of THIS arm, keyed by id.
    Only SUCCESSFUL rows are kept — failed rows are dropped so they get retried.
    Returns {} if no prior artifact exists. Each arm has its own out_path, so
    the two arms never contaminate each other's resume state.
    """
    p = Path(out_path)
    if not p.exists():
        return {}
    prior = json.loads(p.read_text(encoding="utf-8"))
    kept = {
        r["id"]: r
        for r in prior.get("rows", [])
        if r.get("generation_ok")
    }
    logger.info(f"Found prior artifact: reusing {len(kept)} already-succeeded row(s).")
    return kept


def run_capture(eval_set_path: str, out_path: str, *, use_routing: bool) -> None:
    """
    Capture every row in eval_set.json into a single JSON artifact for one arm.

    Idempotent: rows already captured successfully in a prior run of this arm are
    reused as-is (no re-call). Only missing/failed rows hit the API, throttled to
    stay under the free-tier per-minute limit. Re-run freely until all rows green.

    Fails loud if any row is missing a record, or ids collide, before writing.
    """
    eval_set = json.loads(Path(eval_set_path).read_text(encoding="utf-8"))
    already = _load_existing(out_path)

    records = []
    made_live_call = False
    for row in eval_set:
        row_id = row["id"]

        # Reuse a prior successful capture — no API call, no throttle.
        if row_id in already:
            records.append(already[row_id])
            logger.info(f"Skipped {row_id} (already captured).")
            continue

        # Throttle BEFORE each live call except the first, so spacing only
        # applies between real API hits.
        if made_live_call:
            time.sleep(THROTTLE_SECONDS)

        record = capture_row(row["query"], row_id, use_routing=use_routing)
        made_live_call = True
        records.append(record)
        logger.info(
            f"Captured {row_id} "
            f"(ok={record['generation_ok']}, route={record['route']}, mode={record['mode']})."
        )

    # Fail-loud BEFORE writing: every eval row must produce exactly one record.
    # A count mismatch means a row was silently lost — a partial artifact would
    # break the N + gated-out = total reconciliation downstream with no obvious
    # cause.
    if len(records) != len(eval_set):
        raise RuntimeError(
            f"Capture incomplete: {len(records)} records for {len(eval_set)} "
            f"eval rows. Refusing to write a partial artifact."
        )

    # Ids are the join key faithfulness uses against the recall artifact — a
    # duplicate would silently overwrite when either side builds its lookup.
    ids = [r["id"] for r in records]
    if len(set(ids)) != len(ids):
        raise RuntimeError("Duplicate ids in capture records — ids must be unique.")

    artifact = {
        "use_routing": use_routing,          # which arm this artifact is
        "k_value": config.RETRIEVAL_TOP_K,   # the true k retrieved with (=3, matches baseline)
        "row_count": len(records),
        "rows": records,
    }

    Path(out_path).write_text(json.dumps(artifact, indent=2), encoding="utf-8")

    ok = sum(1 for r in records if r["generation_ok"])
    skipped = sum(1 for r in records if r.get("route") == "skip")
    logger.info(
        f"Wrote {out_path}: {len(records)} rows "
        f"({ok} generated, {len(records) - ok} failed, {skipped} skip-routed)."
    )
    if ok < len(records):
        still_failed = sorted(r["id"] for r in records if not r["generation_ok"])
        logger.warning(
            f"{len(records) - ok} row(s) still failed: {still_failed}. "
            f"Re-run to retry only these (succeeded rows are reused)."
        )


if __name__ == "__main__":
    # One flag picks the arm and its output file. Run twice for the full A/B:
    #   python generation_capture.py              # blind baseline
    #   python generation_capture.py --use-routing # agentic
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--use-routing",
        action="store_true",
        help="agentic arm (route decides corpus). Omit for the blind baseline.",
    )
    args = ap.parse_args()

    out_path = (
        "generation_capture_routed.json" if args.use_routing else "generation_capture.json"
    )
    logger.info(
        f"Capture arm: use_routing={args.use_routing} -> {out_path} "
        f"(k={config.RETRIEVAL_TOP_K})"
    )

    run_capture(str(config.EVAL_SET_PATH), out_path, use_routing=args.use_routing)

    # MLflow is intentionally NOT used here. Capture produces no metrics — only
    # answers + context — so it writes a plain JSON artifact to disk and nothing
    # more. The faithfulness step owns the single MLflow run and logs this file
    # into it as the input it gated against.