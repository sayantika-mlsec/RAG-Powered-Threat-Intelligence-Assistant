# Produces the artifact the faithfulness step (Issue 11) consumes: for every eval
# row, the generated answer plus the EXACT context the LLM saw (post-truncation,
# via result.context_used), keyed on `id`. Runs once per arm; faithfulness reads
# this and makes zero retrieval or generation calls — guaranteeing the judged
# context matches the context behind each answer.
#
# A/B CHANGE: capture drives app.run_pipeline instead of calling
# semantic_search + generate_answer directly. use_routing is the primary
# variable between the blind and routed arms, so both are byte-identical
# except the flag — the single-variable discipline the whole routing branch
# exists to preserve.
#
# ARM COLLAPSE (2026-07-22, confirmed against app.py's current run_pipeline):
# run_pipeline() no longer accepts use_confidence_gate at all — the confidence
# gate's retirement removed the three-arm model this script used to support.
# use_routing=True now ALWAYS runs the full retrieve_for_route() pipeline
# (exact-match -> merge/rerank -> fragmentation resolution); there is no more
# "routed but plain semantic_search" mode in the live app. Per run_pipeline's
# own docstring: that comparison still exists in eval_retrieval.py for A/B
# measurement, but not here.
#
# So there are only TWO arms now, not three:
#     use_routing=False -> blind baseline (whole store) -> generation_capture.json
#     use_routing=True  -> full current pipeline          -> generation_capture_routed.json
#
# ⚠️ STALENESS WARNING, not just a naming issue: if generation_capture_routed.json
# already exists locally from BEFORE this collapse, its rows were captured via
# plain semantic_search (the old "routed but ungated" behavior) — a code path
# run_pipeline can no longer produce. REQUIRED_ROW_KEYS will NOT catch this,
# because the schema is unchanged; only which function produced the rows
# changed. _load_existing() would silently reuse those old rows as if they
# came from the current pipeline. Same risk generation_capture_gated.json had
# — treat any pre-existing generation_capture_routed.json as suspect and
# capture fresh via --out-path rather than trusting the default filename,
# same as was done for the gated arm's stale file this session.
#
# The blind arm alone is genuinely untouched by any of this — it never called
# retrieve_for_route() before or after the gate's retirement, so
# generation_capture.json remains valid without re-running.
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
import os
import tempfile
import time
from pathlib import Path
from dotenv import load_dotenv

# Load .env HERE, explicitly, before importing app. app.py also calls
# load_dotenv() internally, and today that happens to run before config gets
# imported (via gemini_client) inside app.py's own init — so this currently
# "works" without this line. But that correctness is borrowed from app.py's
# internal import order, not owned by this script. If app.py's imports ever
# get reordered — a plausible, innocent refactor — this script breaks with a
# KeyError three files away from the line that actually needs the fix. Calling
# load_dotenv() again here is cheap (idempotent, override=True just re-applies
# the same values) and makes this script's correctness self-contained instead
# of borrowed.
load_dotenv(dotenv_path=Path(__file__).parent / ".env", override=True)

# run_pipeline is the single shared entry point (same function the Gradio UI
# calls). Importing it triggers app.py's module-load init of DB / ANALYZER /
# ROUTER_CLIENT, so we do NOT instantiate our own here — capture and the live
# app run through byte-identical retrieval + generation + routing code.
# run_pipeline's signature, as of the gate's retirement, is
# (query, *, use_routing, n_results=N_RESULTS) — NO use_confidence_gate
# parameter. Capture pins n_results to config.RETRIEVAL_TOP_K (k=3) so all
# arms match the pinned baseline capture (which used k=3), not the live UI
# default of N_RESULTS=5.
from app import run_pipeline
import config

# Seconds to sleep between live LLM calls. Tuned for AI Studio's ~10 RPM
# free-tier ceiling. Now on Vertex AI (project-billed, higher per-minute
# limits) — this spacing is safely conservative rather than load-bearing, but
# left as-is until the actual Vertex quota ceiling is confirmed in the Cloud
# Console. Lower it later, deliberately, not by accident.
#
# NOTE — the routed arm makes up to 3 live calls per row when retrieval falls
# through to the rewrite step inside retrieve_for_route() (route decision,
# then rewrite's own LLM call, then generation), vs. 1 for the blind arm.
# THROTTLE_SECONDS is only enforced BETWEEN rows here, not between the calls
# within a single row — retrieve_for_route's rewrite step accepts a
# throttle_fn of its own, left as None from run_pipeline (see app.py). If
# rate-limit errors show up on the routed arm specifically, that's the spot to
# add spacing, not here.
THROTTLE_SECONDS = 6

# Manually-set stamp identifying which version of retrieve_for_route() this
# capture ran against. Not automated versioning — bump by hand whenever
# retrieval logic changes materially (a new merge/rerank revision, a new
# scoring-level fix, etc.), so a cached row's provenance is visible without
# reverse-engineering git history or re-deriving it from timestamps. Written
# on every row this script produces, including blind-arm rows, even though
# the blind arm bypasses retrieve_for_route() entirely — kept for schema
# uniformity across both arms' output files.
#
# Deliberately NOT added to REQUIRED_ROW_KEYS below: doing so would force a
# full regen of the blind arm's artifact too, even though the blind arm's
# underlying retrieval logic (plain semantic_search) didn't change with this
# fix. That would be wasted API spend for zero signal. Staleness on the
# routed arm specifically is handled by writing to a new --out-path rather
# than by this check reusing/rejecting rows.
RETRIEVAL_SNAPSHOT = "fragmentation_fix_2026-07-21"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
)
logger = logging.getLogger(__name__)


def capture_row(
    query: str,
    row_id: str,
    *,
    use_routing: bool,
) -> dict:
    """
    Retrieve + generate for one eval row via run_pipeline. Records the answer and
    the post-truncation context the LLM actually saw (result.context_used), plus
    the route taken and the result mode — both needed for downstream gating.

    NOTE: run_pipeline no longer accepts a use_confidence_gate parameter (see
    module docstring, ARM COLLAPSE) — when use_routing=True, it always runs
    the full current retrieve_for_route() pipeline. Nothing to forward here
    beyond use_routing itself.

    Three outcomes, all recorded (never dropped):
      - retrieved success : generation_ok=True, mode='retrieved', real context
      - skip success      : generation_ok=True, mode='no_retrieval', EMPTY context
                            (route='skip' — gated out of retrieval AND faithfulness
                            scoring downstream; a no-claims answer trivially passes
                            grounding, so it must not enter the faithfulness set)
      - failure           : generation_ok=False, answer null, flagged

    `tier` is recorded on every row (None on the blind arm, same as `route`) —
    this is what the cost/latency-per-tier table groups by downstream.

    `retrieval_snapshot` is recorded on every row (see module-level constant,
    above) — provenance marker, not used for any gating logic here.

    `latency_seconds`/`input_tokens`/`output_tokens`/`thinking_tokens` come
    straight from AnalysisResult (see threat_analyzer.py) — None on the
    no_retrieval (skip) path, since that path never calls ThreatAnalyzer at
    all. thinking is deliberately left ON for these calls (not disabled like
    every other Gemini call site in this project), so output_tokens and
    thinking_tokens are tracked as separate fields even though Google bills
    them together — the eventual table can show the split.
    """
    result, _search_results, route_value, tier_value = run_pipeline(
        query,
        use_routing=use_routing,
        n_results=config.RETRIEVAL_TOP_K,
    )

    usage_fields = {
        "latency_seconds": getattr(result, "latency_seconds", None),
        "input_tokens": getattr(result, "input_tokens", None),
        "output_tokens": getattr(result, "output_tokens", None),
        "thinking_tokens": getattr(result, "thinking_tokens", None),
    }

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
            "tier": tier_value,                            # None on the blind arm
            "retrieval_snapshot": RETRIEVAL_SNAPSHOT,
            "generated_answer": result.answer,
            "retrieved_context": retrieved_context,       # post-truncation / empty on skip
            "citations": result.source_citations,
            "error": None,
            **usage_fields,
        }

    logger.warning(f"Generation failed for {row_id}: {result.error}")
    return {
        "id": row_id,
        "query": query,
        "generation_ok": False,
        "mode": getattr(result, "mode", "retrieved"),
        "route": route_value,
        "tier": tier_value,
        "retrieval_snapshot": RETRIEVAL_SNAPSHOT,
        "generated_answer": None,
        "retrieved_context": [],
        "citations": [],
        "error": result.error,
        **usage_fields,
    }


# The full set of keys every row this script writes is expected to have.
# Used to detect rows captured under an OLDER (narrower) version of this
# script's record schema — e.g. rows captured before the "tier" field was
# added. Keeping this explicit means a future field addition makes old rows
# fail this check automatically, instead of relying on someone to remember
# to delete stale artifacts by hand every time the schema grows.
#
# retrieval_snapshot is deliberately NOT included here — see the constant's
# own comment, above, for why. NOTE: this check catches SCHEMA staleness
# (missing fields) but NOT CODE-PATH staleness (same fields, produced by a
# retired function) — see the module docstring's STALENESS WARNING regarding
# any pre-existing generation_capture_routed.json from before the gate's
# retirement. This check cannot and does not catch that case.
REQUIRED_ROW_KEYS = {
    "id", "query", "generation_ok", "mode", "route", "tier",
    "generated_answer", "retrieved_context", "citations", "error",
    "latency_seconds", "input_tokens", "output_tokens", "thinking_tokens",
}


def _load_existing(out_path: str) -> dict[str, dict]:
    """
    Load already-captured rows from a prior run of THIS arm, keyed by id.

    A row is reused only if BOTH:
      - generation_ok is True (a real failure always gets retried)
      - it has every key in REQUIRED_ROW_KEYS (schema-complete)

    The second check exists because a prior artifact may have been captured
    under an older, narrower record schema (e.g. rows written before the
    "tier" field existed). Such a row is generation_ok=True but silently
    missing data downstream code now expects — reusing it as-is would
    produce a mixed artifact where some rows have tier and others don't,
    with nothing flagging that it happened. Treated the same as a failed
    row here: dropped, and regenerated on this run.

    Returns {} if no prior artifact exists, or if the file is corrupt (e.g. a
    torn write from a hard crash) — treated the same as "nothing captured yet"
    rather than crashing the whole script before a single row is attempted.
    Each arm has its own out_path, so no arm ever contaminates another arm's
    resume state.
    """
    p = Path(out_path)
    if not p.exists():
        return {}
    try:
        prior = json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        logger.warning(f"{out_path} is corrupt/unreadable — starting fresh for this arm.")
        return {}

    kept: dict[str, dict] = {}
    stale_schema_ids: list[str] = []
    for r in prior.get("rows", []):
        if not r.get("generation_ok"):
            continue
        if not REQUIRED_ROW_KEYS.issubset(r.keys()):
            stale_schema_ids.append(r.get("id", "<no id>"))
            continue
        kept[r["id"]] = r

    logger.info(
        f"Found prior artifact: reusing {len(kept)} already-succeeded, "
        f"schema-complete row(s)."
    )
    if stale_schema_ids:
        missing_example = sorted(REQUIRED_ROW_KEYS)
        logger.warning(
            f"{len(stale_schema_ids)} row(s) captured under an older record "
            f"schema (missing one or more of {missing_example}) — will be "
            f"regenerated this run: {sorted(stale_schema_ids)}"
        )
    return kept


def _write_artifact(
    out_path: str,
    records_by_id: dict,
    use_routing: bool,
) -> None:
    """
    Persist current progress atomically. Writes to a temp file in the same
    directory, then os.replace()'s it over out_path — os.replace is atomic on
    both POSIX and Windows, so a crash mid-write can never leave a corrupt,
    unparseable artifact for the next run's _load_existing to choke on.
    Safe to call after every row.
    """
    artifact = {
        "use_routing": use_routing,
        "retrieval_snapshot": RETRIEVAL_SNAPSHOT,
        "k_value": config.RETRIEVAL_TOP_K,
        "row_count": len(records_by_id),
        "rows": list(records_by_id.values()),
    }
    out_dir = Path(out_path).parent or Path(".")
    fd, tmp_path = tempfile.mkstemp(dir=out_dir, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(artifact, f, indent=2)
        os.replace(tmp_path, out_path)  # atomic swap — old file or new file, never half of either
    except Exception:
        os.unlink(tmp_path)  # don't leave a stray .tmp lying around
        raise


def run_capture(
    eval_set_path: str,
    out_path: str,
    *,
    use_routing: bool,
) -> None:
    """
    Capture every row in eval_set.json into a single JSON artifact for one arm.

    Idempotent: rows already captured successfully in a prior run of this arm are
    reused as-is (no re-call). Only missing/failed rows hit the API, throttled to
    stay under the free-tier per-minute limit. Re-run freely until all rows green.

    Fails loud on duplicate ids BEFORE any API call. Checkpoints atomically after
    every row, so a crash mid-run loses at most one row's spend, not the session.
    """
    eval_set = json.loads(Path(eval_set_path).read_text(encoding="utf-8"))

    # Fail loud BEFORE spending a single API call: records_by_id below is keyed
    # by id, so a duplicate id would silently overwrite one row's captured
    # answer with another's.
    ids = [row["id"] for row in eval_set]
    if len(set(ids)) != len(ids):
        dupes = sorted({i for i in ids if ids.count(i) > 1})
        raise RuntimeError(f"Duplicate ids in eval_set.json: {dupes}. Fix before running.")

    records_by_id = _load_existing(out_path)  # {} or prior successful rows
    made_live_call = False

    for row in eval_set:
        row_id = row["id"]

        if row_id in records_by_id:
            logger.info(f"Skipped {row_id} (already captured).")
            continue

        if made_live_call:
            time.sleep(THROTTLE_SECONDS)

        # ISOLATE the failure: one bad row must not kill the whole run, and
        # must not lose every row already captured this session.
        try:
            record = capture_row(
                row["query"], row_id,
                use_routing=use_routing,
            )
        except Exception as e:
            logger.error(f"Row {row_id} raised {e!r} — recording as failed, continuing.")
            record = {
                "id": row_id,
                "query": row["query"],
                "generation_ok": False,
                "mode": None,
                "route": None,
                "tier": None,
                "retrieval_snapshot": RETRIEVAL_SNAPSHOT,
                "generated_answer": None,
                "retrieved_context": [],
                "citations": [],
                "error": str(e),
                "latency_seconds": None,
                "input_tokens": None,
                "output_tokens": None,
                "thinking_tokens": None,
            }

        made_live_call = True
        records_by_id[row_id] = record
        logger.info(
            f"Captured {row_id} (ok={record['generation_ok']}, route={record['route']}, "
            f"tier={record['tier']}, mode={record['mode']})."
        )

        # CHECKPOINT after every row. Worst case on a crash: you lose the one
        # row in flight, not the whole session's spend.
        _write_artifact(out_path, records_by_id, use_routing)

    # Final integrity check — checked against what's actually on disk/in memory.
    missing = [row["id"] for row in eval_set if row["id"] not in records_by_id]
    if missing:
        raise RuntimeError(f"Capture incomplete: missing rows {missing}. Re-run to fill gaps.")

    ok = sum(1 for r in records_by_id.values() if r["generation_ok"])
    skipped = sum(1 for r in records_by_id.values() if r.get("route") == "skip")
    logger.info(
        f"Wrote {out_path}: {len(records_by_id)} rows "
        f"({ok} generated, {len(records_by_id) - ok} failed, {skipped} skip-routed)."
    )
    if ok < len(records_by_id):
        still_failed = sorted(rid for rid, r in records_by_id.items() if not r["generation_ok"])
        logger.warning(
            f"{len(records_by_id) - ok} row(s) still failed: {still_failed}. "
            f"Re-run to retry only these (succeeded rows are reused)."
        )


if __name__ == "__main__":
    # ONE flag now picks the arm (see module docstring, ARM COLLAPSE);
    # --out-path overrides its default output filename explicitly.
    #   python generation_capture.py                    # blind baseline
    #   python generation_capture.py --use-routing       # full current pipeline
    #
    # --out-path example — capturing against a new retrieval snapshot without
    # touching a prior arm's artifact from an older pipeline version:
    #   python generation_capture.py --use-routing \
    #       --out-path generation_capture_post_frag.json
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--use-routing",
        action="store_true",
        help="full current pipeline — route_query() picks a corpus, then "
             "retrieve_for_route() (exact-match -> merge/rerank -> "
             "fragmentation resolution). Omit for the blind baseline.",
    )
    ap.add_argument(
        "--out-path",
        default=None,
        help="Override the default output filename for this arm. Useful when "
             "capturing against a new retrieval snapshot without renaming or "
             "overwriting a prior artifact from an older pipeline version.",
    )
    args = ap.parse_args()

    out_path = "generation_capture_routed.json" if args.use_routing else "generation_capture.json"

    if args.out_path:      # explicit override wins over the flag-derived default
        out_path = args.out_path

    logger.info(
        f"Capture arm: use_routing={args.use_routing} -> {out_path} "
        f"(k={config.RETRIEVAL_TOP_K}, retrieval_snapshot={RETRIEVAL_SNAPSHOT})"
    )

    run_capture(
        str(config.EVAL_SET_PATH), out_path,
        use_routing=args.use_routing,
    )

    # MLflow is intentionally NOT used here. Capture produces no metrics — only
    # answers + context — so it writes a plain JSON artifact to disk and nothing
    # more. The faithfulness step owns the single MLflow run and logs this file
    # into it as the input it gated against.