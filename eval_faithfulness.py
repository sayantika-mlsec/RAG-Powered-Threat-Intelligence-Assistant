# Issue 11 deliverable: faithfulness scoring via Gemini-as-judge.
#
# Reads two artifacts, gates, judges, reconciles, logs — in ONE MLflow run:
#   1. recall artifact     (from eval_retrieval.py)      -> eligibility per id
#   2. generation_capture  (from generation_capture.py)  -> answer + context per id
#
# Faithfulness = do the generated answer's claims appear in the retrieved
# context? Judge compares generated-vs-context ONLY. Gold answers are NOT an
# input. Correctness is a separate, out-of-scope concern.
#
# Gate: score ONLY rows where eligible == true (recall@3 > 0). Eligibility is
# READ from the recall artifact — retrieval is never recomputed here.
#
# Judge output format: STRUCTURED JSON {"score": int, "reason": str}. The reason
# is kept for failure analysis; the score is the metric. (To switch to bare-int
# scoring, see _parse_score and JUDGE_SYSTEM_INSTRUCTION — both are marked.)

import os
import re
import json
import time
import logging
import tempfile
from pathlib import Path

from google.genai import types as genai_types
from dotenv import load_dotenv
import mlflow
from mlflow.artifacts import download_artifacts

import config

from gemini_client import get_client

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
)
logger = logging.getLogger(__name__)

load_dotenv(dotenv_path=Path(__file__).parent / ".env", override=True)

# Same per-minute-limit throttle as capture. Judge makes one call per eligible
# row, so spacing matters for the same reason it did in capture.
THROTTLE_SECONDS = 6

GROUP_A_TOTAL = 15   # reconciliation invariant: N_scored + gated_out == 15

# Recall artifact lives in MLflow under the pinned baseline run (Issue 1).
# Read by run id — MLflow resolves its physical location under mlartifacts/,
# so no file is moved and no brittle path is hardcoded. The run id is the
# real lineage: this faithfulness score is provably tied to the run that
# produced eligibility.
RECALL_BASELINE_RUN_ID = os.getenv("RECALL_BASELINE_RUN_ID")
if not RECALL_BASELINE_RUN_ID:
    raise ValueError(
        "RECALL_BASELINE_RUN_ID not set in .env — cannot locate the recall "
        "artifact or record lineage."
    )

# Artifact DIRECTORY within the run (not the file). Issue 9 logged the JSON
# under a temp filename, so we target the folder and find the single JSON
# inside rather than hardcoding a non-deterministic name.
RECALL_ARTIFACT_DIR = "per_row_metrics"

# Resumable scoring: per-row scores are persisted here as they're judged, so a
# quota cutoff mid-run isn't fatal — re-running skips already-scored rows.
# This is SCRATCH (gitignore it). MLflow logging fires only when all N rows are
# scored, so a partial scratch file never produces a partial logged mean.
SCORES_SCRATCH_PATH = "faithfulness_scores.partial.json"

JUDGE_SYSTEM_INSTRUCTION = (
    "You are a strict faithfulness grader for a retrieval-augmented threat-"
    "intelligence assistant. You are given an ANSWER and the CONTEXT it was "
    "supposed to be grounded in. Your ONLY job is to judge whether every factual "
    "claim in the ANSWER appears in the CONTEXT.\n"
    "\n"
    "You are NOT judging whether the answer is correct, helpful, or well written. "
    "You are judging ONLY grounding: does each claim trace to the provided "
    "context? A claim that is true in the real world but absent from the context "
    "is UNFAITHFUL. Do not use outside knowledge.\n"
    "\n"
    "Score on this 1-5 rubric:\n"
    "5 - Fully grounded. Every factual claim traces to the context.\n"
    "4 - Grounded, minor slack. All substantive claims sourced; only trivial "
    "connective phrasing is unsourced. Nothing contradicts the context.\n"
    "3 - One unsupported factual addition. Core answer sourced, but at least one "
    "factual claim is absent from the context (additive, not contradictory).\n"
    "2 - Partially grounded. A central claim is unsupported, or there are multiple "
    "unsupported additions.\n"
    "1 - Unfaithful. A central claim contradicts the context, or the answer is "
    "largely fabricated (invented IDs/relationships not in the context).\n"
    "\n"
    "Worked examples (CONTEXT describes technique T1210, Exploitation of Remote "
    "Services):\n"
    "- ANSWER: 'T1210 exploits remote services; commonly seen in intrusions.' "
    "The second clause is trivial connective framing, not a load-bearing fact -> 4.\n"
    "- ANSWER: 'T1210 exploits remote services and was used in the WannaCry "
    "attack.' The WannaCry claim is a specific fact absent from the context -> 3.\n"
    "- ANSWER: 'This maps to T1566, Phishing.' The claim contradicts the context "
    "(wrong technique) -> 1.\n"
    "\n"
    # ── OUTPUT FORMAT (switch point: bare-int vs structured JSON) ──
    "Respond with ONLY a JSON object and nothing else — no markdown, no code "
    "fences, no commentary. Exactly this shape:\n"
    '{\"score\": <integer 1-5>, \"reason\": \"<one short sentence>\"}'
)


# ── Artifact loading ──────────────────────────────────────────────────────────

def load_recall_artifact(run_id: str, artifact_dir: str = RECALL_ARTIFACT_DIR) -> dict[str, bool]:
    """
    Pull the recall artifact from its MLflow run and return {id: eligible}.

    Targets the artifact DIRECTORY (not the file) because Issue 1 logged the
    JSON under a non-deterministic temp filename. Downloads the folder, then
    reads the single JSON inside — fail-loud if zero or more than one is found.
    """
    local_dir = download_artifacts(run_id=run_id, artifact_path=artifact_dir)
    json_files = list(Path(local_dir).glob("*.json"))
    if len(json_files) != 1:
        raise RuntimeError(
            f"Expected exactly one JSON in '{artifact_dir}' of run {run_id}, "
            f"found {len(json_files)}: {[f.name for f in json_files]}"
        )
    data = json.loads(json_files[0].read_text(encoding="utf-8"))
    return {row["id"]: bool(row["eligible"]) for row in data["rows"]}


def load_capture_artifact(path: str) -> dict[str, dict]:
    """Return {id: capture_row} from the generation-capture artifact."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return {row["id"]: row for row in data["rows"]}


def load_partial_scores(path: str) -> dict[str, dict]:
    """
    Load already-judged rows from a prior interrupted run, keyed by id.
    Returns {} if no scratch file exists, OR if it exists but is corrupt (e.g.
    a torn write from a hard kill) — treated the same as "nothing judged yet"
    rather than crashing resume before a single row is attempted.
    """
    p = Path(path)
    if not p.exists():
        return {}
    try:
        rows = json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        logger.warning(f"{path} is corrupt — starting fresh (prior judged rows lost).")
        return {}
    scored = {r["id"]: r for r in rows}
    logger.info(f"Resuming: {len(scored)} row(s) already judged in scratch file.")
    return scored


def save_partial_scores(path: str, scored: dict[str, dict]) -> None:
    """
    Persist judged rows after each call, atomically. Writes to a temp file in
    the same directory, then os.replace()'s it over path — atomic on both
    POSIX and Windows, so a crash mid-write can't leave a half-written scratch
    file for the next resume to choke on. The whole point of checkpointing
    after every call is defeated if the checkpoint itself can be corrupted by
    the same crash it's meant to survive.
    """
    out_dir = Path(path).parent or Path(".")
    fd, tmp_path = tempfile.mkstemp(dir=out_dir, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(list(scored.values()), f, indent=2)
        os.replace(tmp_path, path)
    except Exception:
        os.unlink(tmp_path)
        raise


# ── Judge ─────────────────────────────────────────────────────────────────────

def _build_judge_prompt(answer: str, context_chunks: list[str]) -> str:
    """
    Assemble the judge input: the answer to grade + the context to grade against.
    Tagged sections so the judge cannot confuse answer-text with context-text.
    """
    context_text = "\n\n---\n\n".join(context_chunks)
    return (
        "<context>\n"
        f"{context_text}\n"
        "</context>\n"
        "<answer>\n"
        f"{answer}\n"
        "</answer>\n"
        "Grade the ANSWER's faithfulness to the CONTEXT using the rubric. "
        "Return ONLY the JSON object."
    )


def _safe_extract_text(response) -> tuple[str | None, str | None]:
    """
    Extract text from a new-SDK (google.genai) response, distinguishing failure
    modes. Returns (text, error_reason) — exactly one is None.

    MAX_TOKENS is treated as a FAILURE here (unlike the generator): a truncated
    reply is unparseable JSON for the judge, so it must not pass as success.
    """
    if not response.candidates:
        return None, "NO_CANDIDATES"
    candidate = response.candidates[0]
    finish_reason = candidate.finish_reason.name if candidate.finish_reason else "UNKNOWN"
    if finish_reason == "SAFETY":
        return None, "CANDIDATE_SAFETY_FILTERED"
    if finish_reason == "MAX_TOKENS":
        return None, "TRUNCATED_MAX_TOKENS"   # judge JSON cut off — raise max_output_tokens
    if finish_reason != "STOP":
        return None, f"UNEXPECTED_FINISH:{finish_reason}"
    text = response.text
    if not text:
        return None, "EMPTY_TEXT"
    return text, None


def _parse_score(raw: str) -> tuple[int, str]:
    """
    Parse the judge's structured-JSON reply into (score, reason).
    Raises ValueError on anything unparseable or out of range — a silently
    coerced bad parse would corrupt the metric. NEVER default to a middle value.

    (Bare-int switch point: to score on a bare integer instead of JSON, replace
    the JSON parse below with an int extraction and return (int, "").)
    """
    cleaned = raw.strip()
    # Tolerate accidental code fences the model sometimes adds.
    cleaned = re.sub(r"^```(?:json)?|```$", "", cleaned, flags=re.MULTILINE).strip()
    try:
        obj = json.loads(cleaned)
    except json.JSONDecodeError as e:
        raise ValueError(f"Judge reply was not valid JSON: {raw!r}") from e

    if "score" not in obj:
        raise ValueError(f"Judge reply missing 'score': {raw!r}")

    score = obj["score"]
    if not isinstance(score, int) or score < 1 or score > 5:
        raise ValueError(f"Judge score not an int in 1..5: {raw!r}")

    return score, str(obj.get("reason", "")).strip()


def judge_faithfulness(client, model_name: str, answer: str, context_chunks: list[str]) -> tuple[int, str]:
    """One judge call (new google.genai SDK) for one row. Returns (score, reason) or raises."""
    prompt = _build_judge_prompt(answer, context_chunks)
    response = client.models.generate_content(
        model=model_name,
        contents=prompt,
        config=genai_types.GenerateContentConfig(
            system_instruction=JUDGE_SYSTEM_INSTRUCTION,
            temperature=0.0,            # reproducible scoring
            max_output_tokens=256,      # JSON is tiny once thinking is off
            # Disable thinking: 2.5-flash thinks by default and thinking tokens
            # eat the output budget, truncating the JSON. The judge applies a
            # rubric — no chain-of-thought needed.
            thinking_config=genai_types.ThinkingConfig(thinking_budget=0),
        ),
    )
    text, error_reason = _safe_extract_text(response)
    if error_reason:
        # Do NOT swallow — a failed judge call must not silently become a score.
        raise RuntimeError(f"Judge call failed: {error_reason}")
    return _parse_score(text)


# ── Main ──────────────────────────────────────────────────────────────────────

def run_faithfulness(recall_run_id: str, capture_path: str) -> None:
    eligibility = load_recall_artifact(recall_run_id)   # {id: eligible}, pulled by run id
    captures    = load_capture_artifact(capture_path)   # {id: row}

    eligible_ids = [rid for rid, ok in eligibility.items() if ok]
    gated_out    = [rid for rid, ok in eligibility.items() if not ok]

    # RECONCILE before any scoring: the two groups must cover Group A exactly.
    # If this fails the artifacts disagree about the eval set — stop, don't score
    # against an inconsistent denominator.
    if len(eligible_ids) + len(gated_out) != GROUP_A_TOTAL:
        raise RuntimeError(
            f"Reconciliation failed: {len(eligible_ids)} eligible + "
            f"{len(gated_out)} gated-out != {GROUP_A_TOTAL} (Group A total)."
        )

    # Every eligible row MUST have a successful capture to judge. A missing or
    # failed capture for an eligible id is a hard error.
    for rid in eligible_ids:
        cap = captures.get(rid)
        if cap is None or not cap.get("generation_ok"):
            raise RuntimeError(
                f"Eligible row {rid} has no successful capture — cannot score. "
                f"Re-run generation_capture.py."
            )

    N = len(eligible_ids)

    mlflow.set_experiment(config.MLFLOW_EXPERIMENT_NAME)

    # EMPTY-ELIGIBLE PATH: if nothing is eligible, do NOT compute a numeric score.
    if N == 0:
        logger.warning("No eligible rows (N=0). Logging null faithfulness, no score.")
        with mlflow.start_run(run_name="faithfulness_eval"):
            mlflow.log_metric("n_eligible", 0)
            mlflow.log_metric("n_gated_out", len(gated_out))
            mlflow.set_tag("faithfulness_mean", "null_no_eligible_rows")
            mlflow.log_param("recall_run_id", recall_run_id)   # real lineage to baseline
            mlflow.log_param("capture_artifact", capture_path)
        return

    # ── Judge each eligible row ───────────────────────────────────────────────
    judge_client = get_client()
    judge_model_name = os.getenv("GEMINI_MODEL_NAME", "gemini-2.5-flash")

    # Resume from any prior interrupted run — only judge rows not already scored.
    scored = load_partial_scores(SCORES_SCRATCH_PATH)
    # Defensive: drop any scratch entries that aren't in this run's eligible set
    # (e.g. stale scratch from a different eval set), so resume can't smuggle in
    # rows that don't belong.
    scored = {rid: r for rid, r in scored.items() if rid in eligible_ids}

    made_live_call = False
    for rid in eligible_ids:
        if rid in scored:
            logger.info(f"Skipped {rid} (already judged).")
            continue

        # Throttle before each LIVE judge call except the first this run.
        if made_live_call:
            time.sleep(THROTTLE_SECONDS)

        cap = captures[rid]
        score, reason = judge_faithfulness(
            judge_client, judge_model_name, cap["generated_answer"], cap["retrieved_context"]
        )
        made_live_call = True
        scored[rid] = {"id": rid, "faithfulness": score, "reason": reason}
        save_partial_scores(SCORES_SCRATCH_PATH, scored)   # persist after each call, atomically
        logger.info(f"Judged {rid}: faithfulness={score}")

    # Only here, with all N scored, do we compute the mean and log to MLflow.
    # A partial scratch file never reaches this point — the loop above must
    # complete the full eligible set first.
    per_row = [scored[rid] for rid in eligible_ids]
    mean_faithfulness = sum(r["faithfulness"] for r in per_row) / N

    # ── Log: ONE run, score never bare ────────────────────────────────────────
    # Contract: score travels with N, gated-out count, and lineage (which
    # artifacts it gated against). N + gated-out reconciles to GROUP_A_TOTAL.
    with mlflow.start_run(run_name="faithfulness_eval"):
        mlflow.log_metric("faithfulness_mean", mean_faithfulness)
        mlflow.log_metric("n_eligible", N)
        mlflow.log_metric("n_gated_out", len(gated_out))
        # recall_run_id IS the lineage now — the recall artifact was pulled from
        # this exact run, so the score is provably tied to the baseline that
        # produced eligibility (not a loose file we have to trust).
        mlflow.log_param("recall_run_id", recall_run_id)
        mlflow.log_param("capture_artifact", capture_path)
        mlflow.log_param("judge_model", judge_model_name)
        mlflow.log_dict(
            {
                "faithfulness_mean": mean_faithfulness,
                "n_eligible": N,
                "n_gated_out": len(gated_out),
                "gated_out_ids": sorted(gated_out),
                "group_a_total": GROUP_A_TOTAL,
                "rows": per_row,
            },
            "faithfulness/per_row.json",
        )
        # Log the capture file as the input this run gated against (lineage).
        mlflow.log_artifact(capture_path, artifact_path="faithfulness_inputs")

    # Run logged successfully — clear the scratch file so the NEXT run starts
    # fresh rather than resuming from now-stale scores.
    Path(SCORES_SCRATCH_PATH).unlink(missing_ok=True)

    logger.info(
        f"Faithfulness mean={mean_faithfulness:.3f} over N={N} eligible "
        f"({len(gated_out)} gated out, reconciles to {GROUP_A_TOTAL})."
    )


if __name__ == "__main__":
    # Recall comes from MLflow by run id (RECALL_BASELINE_RUN_ID, from .env,
    # validated at module load). Capture is the local artifact from this issue.
    #
    # ⚠️ TRAP: this is hardcoded to the BLIND arm. Scoring the
    # ROUTED arm means changing this to "generation_capture_routed.json" AND
    # repointing RECALL_BASELINE_RUN_ID in .env to the routed retrieval run —
    # miss either one and routed answers silently gate against blind eligibility.
    CAPTURE_ARTIFACT = "generation_capture.json"
    run_faithfulness(RECALL_BASELINE_RUN_ID, CAPTURE_ARTIFACT)