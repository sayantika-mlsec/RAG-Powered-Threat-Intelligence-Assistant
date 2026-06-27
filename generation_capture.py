# Produces the artifact the faithfulness step (Issue 11) consumes: for every eval
# row, the generated answer plus the EXACT context the LLM saw (post-truncation,
# via result.context_used), keyed on `id`. Runs once; faithfulness reads this and
# makes zero retrieval or generation calls — guaranteeing the judged context
# matches the context behind each answer.
#
# Scope: capture records ALL rows, including ineligible ones. The eligibility gate
# (recall@3 > 0) lives in the faithfulness step, not here. Rows whose generation
# fails (e.g. no chunks retrieved) are recorded WITH A FLAG, never skipped, so
# every eval row maps to exactly one record.

import json
import logging
import time
from pathlib import Path

from threat_analyzer import ThreatAnalyzer   
from ingest import ThreatIntelDB
import config

# Seconds to sleep between live LLM calls, to stay under the free-tier
# per-minute rate limit. Free tier is ~10 RPM, so ~6s spacing keeps us safe.
# Applied only between rows that actually hit the API — skipped rows cost nothing.
THROTTLE_SECONDS = 6

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
)
logger = logging.getLogger(__name__)


def capture_row(
    analyzer: ThreatAnalyzer,
    db: ThreatIntelDB,
    query: str,
    row_id: str,
) -> dict:
    """
    Retrieve + generate for one eval row. Records the answer and the
    post-truncation context the LLM actually saw (result.context_used).

    Failed generation is recorded with generation_ok=False, NOT dropped —
    faithfulness needs the row to exist even though it can't be scored.
    """
    # 1. Retrieve — raw search_results dict, exactly as the live RAG passes it.
    #    n_results is the param name (not k); pull from config, don't hardcode.
    search_results = db.semantic_search(query, n_results=config.RETRIEVAL_TOP_K)

    # 2. Generate — pass the raw dict; generate_answer does validation,
    #    truncation, prompt-building, and the LLM call internally.
    result = analyzer.generate_answer(query, search_results)

    # 3. Build the record. On success, store the answer + the exact context it
    #    was grounded in (context_used). On failure, flag it, answer stays null.
    if result.success:
        return {
            "id": row_id,
            "query": query,
            "generation_ok": True,
            "generated_answer": result.answer,
            "retrieved_context": result.context_used,   # post-truncation — what the LLM saw
            "citations": result.source_citations,        # useful for failure analysis
            "error": None,
        }

    logger.warning(f"Generation failed for {row_id}: {result.error}")
    return {
        "id": row_id,
        "query": query,
        "generation_ok": False,
        "generated_answer": None,
        "retrieved_context": [],
        "citations": [],
        "error": result.error,
    }


def _load_existing(out_path: str) -> dict[str, dict]:
    """
    Load already-captured rows from a prior run, keyed by id.
    Only SUCCESSFUL rows are kept — failed rows are dropped so they get retried.
    Returns {} if no prior artifact exists.
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


def run_capture(eval_set_path: str, out_path: str) -> None:
    """
    Capture every row in eval_set.json into a single JSON artifact.

    Idempotent: rows already captured successfully in a prior run are reused
    as-is (no re-call). Only missing/failed rows hit the API, throttled to stay
    under the free-tier per-minute limit. Re-run freely until all rows are green.

    Fails loud if any row is missing a record, or ids collide, before writing.
    """
    eval_set = json.loads(Path(eval_set_path).read_text(encoding="utf-8"))

    already = _load_existing(out_path)

    analyzer = ThreatAnalyzer()   # instantiate once, reuse across rows
    db = ThreatIntelDB()          # instantiate once, reuse across rows

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

        record = capture_row(analyzer, db, row["query"], row_id)
        made_live_call = True
        records.append(record)
        logger.info(f"Captured {row_id} (generation_ok={record['generation_ok']}).")

    # Fail-loud BEFORE writing: every eval row must produce exactly one record.
    # A count mismatch means a row was silently lost — a partial artifact would
    # make faithfulness under-count and break the N + gated-out = 15
    # reconciliation downstream with no obvious cause.
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
        "k_value": config.RETRIEVAL_TOP_K,
        "row_count": len(records),
        "rows": records,
    }

    Path(out_path).write_text(json.dumps(artifact, indent=2), encoding="utf-8")
    ok = sum(1 for r in records if r["generation_ok"])
    logger.info(
        f"Wrote {out_path}: {len(records)} rows "
        f"({ok} generated, {len(records) - ok} failed)."
    )
    if ok < len(records):
        still_failed = sorted(r["id"] for r in records if not r["generation_ok"])
        logger.warning(
            f"{len(records) - ok} row(s) still failed: {still_failed}. "
            f"Re-run to retry only these (succeeded rows are reused)."
        )


if __name__ == "__main__":
    EVAL_SET_PATH = str(config.EVAL_SET_PATH)
    OUT_PATH = "generation_capture.json"

    run_capture(EVAL_SET_PATH, OUT_PATH)

    # MLflow is intentionally NOT used here. Capture produces no metrics — only
    # answers + context — so it writes a plain JSON artifact to disk and nothing
    # more. Logging a metric-less run would be using MLflow as a file store.
    #
    # The faithfulness step (eval_faithfulness.py) owns the single MLflow run: it
    # logs the faithfulness scores (the actual metrics) AND logs this capture file
    # into that same run as the input it gated against. One run holds the scores
    # and their input together — no second run, no cross-sourcing.