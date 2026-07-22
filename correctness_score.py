# correctness_score.py — correctness-vs-gold scoring via Gemini-as-judge.
#
# Resolves the open design question from the July 22 handoff (Open Design
# Question #1): does q009's judge finding belong to the faithfulness judge
# or to a separate correctness metric? Answer, decided before this file was
# written: correctness metric. Faithfulness's contract ("grounding only, not
# correctness" — see JUDGE_SYSTEM_INSTRUCTION in eval_faithfulness.py) stays
# narrow and untouched. q006 and q009 are NOT special-cased here — they fall
# out of this scorer as ordinary rows, same as every other query.
#
# Correctness = does the candidate answer convey the same substantive facts
# as the hand-written, corpus-verified gold answer? Judge compares
# candidate-vs-gold ONLY. Retrieved context is NOT an input — an ungrounded
# lucky guess that happens to match gold still scores well here; that's a
# faithfulness problem, not a correctness one. The two metrics are meant to
# be read together (see docstring at bottom) — not merged into one.
#
# DELIBERATE DIVERGENCE FROM eval_faithfulness.py's PATTERN:
# Faithfulness gates on eligibility (recall@3 > 0) because grounding is
# undefined without retrieved context. Correctness has no such dependency —
# a refusal caused by bad retrieval is exactly the case this metric needs to
# score, not exclude. So there is NO eligibility gate here: all 15 Group A
# rows are scored unconditionally, every run. N is fixed, not variable.
# Everything else (resumable scratch file, atomic writes, reconciliation
# invariant, MLflow lineage logging) is reused deliberately.

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

import config
from gemini_client import get_client

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
)
logger = logging.getLogger(__name__)

load_dotenv(dotenv_path=Path(__file__).parent / ".env", override=True)

THROTTLE_SECONDS = 6  # same per-minute-limit reasoning as faithfulness_score.py

GROUP_A_TOTAL = 15  # reconciliation invariant: N_scored must == 15, always

# Resumable scoring: same rationale as eval_faithfulness.py. Separate scratch
# file so a correctness run and a faithfulness run can be interrupted/resumed
# independently without colliding. SCRATCH (gitignore it)
SCORES_SCRATCH_PATH = "correctness_scores.partial.json"

CORRECTNESS_JUDGE_SYSTEM_INSTRUCTION = (
    "You are a strict correctness grader for a retrieval-augmented threat-"
    "intelligence assistant. You are given the ORIGINAL QUESTION, a GOLD "
    "ANSWER written and verified by a human against the source corpus, and a "
    "CANDIDATE ANSWER produced by the system. Your ONLY job is to judge "
    "whether the CANDIDATE ANSWER conveys the same correct, substantive "
    "facts as the GOLD ANSWER.\n"
    "\n"
    "You are NOT judging whether the candidate is grounded in any retrieved "
    "context — an ungrounded lucky guess that happens to match the gold "
    "answer is still CORRECT for this metric. You are judging ONLY outcome "
    "correctness: did the user end up knowing the right thing?\n"
    "\n"
    "A refusal or non-answer conveys none of the gold answer's facts, and "
    "scores accordingly low — regardless of whether the refusal was a "
    "reasonable thing for the system to do given what it had access to. "
    "This metric does not evaluate whether a refusal was justified, only "
    "whether the user received the correct information.\n"
    "\n"
    "Score on this 1-5 rubric:\n"
    "5 - Fully correct. Every key fact in GOLD (IDs, causal/sequential "
    "relationships, named mechanisms) appears correctly in CANDIDATE.\n"
    "4 - Correct, minor gap. All key facts present and correct; a "
    "supporting or secondary detail from GOLD is thin or missing, but "
    "nothing stated is wrong.\n"
    "3 - Partially correct. Some key facts from GOLD are present and "
    "correct, but at least one is missing, muddled, or a multi-part answer "
    "is incomplete.\n"
    "2 - Largely incorrect. The core identifying fact (e.g. wrong "
    "technique/CVE ID, wrong direction of a relationship) is wrong, though "
    "the candidate is on-topic.\n"
    "1 - No correct information delivered. The candidate contradicts GOLD "
    "outright, invents facts absent from GOLD, or is a refusal/non-answer "
    "providing none of GOLD's content.\n"
    "\n"
    "Worked examples:\n"
    "- GOLD: 'T1210, Exploitation of Remote Services.' CANDIDATE: 'This is "
    "T1210, Exploitation of Remote Services, used to gain access via a "
    "vulnerable service.' -> 5.\n"
    "- GOLD: 'T1210, Exploitation of Remote Services.' CANDIDATE: 'This is "
    "T1021, Remote Services.' Wrong technique ID entirely -> 1.\n"
    "- GOLD: 'CVE-2020-3161 is in KEV; maps to T1190.' CANDIDATE: 'I do not "
    "have sufficient information to answer this.' No facts delivered, "
    "regardless of whether the refusal was reasonable given what the system "
    "actually retrieved -> 1.\n"
    "\n"
    "Respond with ONLY a JSON object and nothing else — no markdown, no "
    "code fences, no commentary. Exactly this shape:\n"
    '{\"score\": <integer 1-5>, \"reason\": \"<one short sentence>\"}'
)


# ── Eval-set / Group A derivation ──────────────────────────────────────────────

def load_eval_set(path: str) -> dict[str, dict]:
    """Return {id: row} from eval_set.json."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return {row["id"]: row for row in data}


def compute_group_a_ids(eval_rows: dict[str, dict]) -> list[str]:
    """
    Group A = rows with at least one expected ID (technique or CVE) — the
    same definition eval_pipeline.md uses for "Skip Rows" (q014, q017-q020:
    no expected IDs of any kind, never retrieval-scored), inverted. Derived
    from the eval set itself rather than hardcoded, so this doesn't silently
    drift if the eval set grows (see eval_pipeline.md's Deferred Work).

    Fails loud if the derived count isn't 15 — that mismatch means either
    the eval set changed or this rule no longer matches eval_pipeline.md's
    definition, and scoring against a wrong denominator is worse than
    stopping here.
    """
    group_a = [
        rid for rid, row in eval_rows.items()
        if row.get("expected_technique_ids") or row.get("expected_cve_ids")
    ]
    if len(group_a) != GROUP_A_TOTAL:
        raise RuntimeError(
            f"Derived {len(group_a)} Group A row(s), expected {GROUP_A_TOTAL}. "
            f"Eval set may have changed — reconcile before scoring. "
            f"Derived ids: {sorted(group_a)}"
        )
    return sorted(group_a)


def load_capture_artifact(path: str) -> dict[str, dict]:
    """Return {id: capture_row} from the generation-capture artifact."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return {row["id"]: row for row in data["rows"]}


def load_partial_scores(path: str) -> dict[str, dict]:
    """Same resume/corrupt-scratch handling as faithfulness_score.py."""
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
    """Same atomic tempfile + os.replace pattern as faithfulness_score.py."""
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

def _build_judge_prompt(question: str, gold_answer: str, candidate_answer: str) -> str:
    """
    Assemble the judge input. Tags deliberately distinct from faithfulness's
    <context>/<answer> tags — GOLD here is a reference answer, not retrieved
    context, and conflating the two vocabularies risks a copy-pasted judge
    prompt silently comparing the wrong things later.
    """
    return (
        "<question>\n"
        f"{question}\n"
        "</question>\n"
        "<gold_answer>\n"
        f"{gold_answer}\n"
        "</gold_answer>\n"
        "<candidate_answer>\n"
        f"{candidate_answer}\n"
        "</candidate_answer>\n"
        "Grade the CANDIDATE_ANSWER's correctness against the GOLD_ANSWER "
        "using the rubric. Return ONLY the JSON object."
    )


def _safe_extract_text(response) -> tuple[str | None, str | None]:
    """Identical contract to faithfulness_score.py's version — MAX_TOKENS is
    a failure here too, since a truncated reply is unparseable JSON."""
    if not response.candidates:
        return None, "NO_CANDIDATES"
    candidate = response.candidates[0]
    finish_reason = candidate.finish_reason.name if candidate.finish_reason else "UNKNOWN"
    if finish_reason == "SAFETY":
        return None, "CANDIDATE_SAFETY_FILTERED"
    if finish_reason == "MAX_TOKENS":
        return None, "TRUNCATED_MAX_TOKENS"
    if finish_reason != "STOP":
        return None, f"UNEXPECTED_FINISH:{finish_reason}"
    text = response.text
    if not text:
        return None, "EMPTY_TEXT"
    return text, None


def _parse_score(raw: str) -> tuple[int, str]:
    """Identical contract to faithfulness_score.py's version — never
    default to a middle value on a bad parse."""
    cleaned = raw.strip()
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


def judge_correctness(client, model_name: str, question: str, gold_answer: str, candidate_answer: str) -> tuple[int, str]:
    """One judge call for one row. Returns (score, reason) or raises."""
    prompt = _build_judge_prompt(question, gold_answer, candidate_answer)
    response = client.models.generate_content(
        model=model_name,
        contents=prompt,
        config=genai_types.GenerateContentConfig(
            system_instruction=CORRECTNESS_JUDGE_SYSTEM_INSTRUCTION,
            temperature=0.0,
            max_output_tokens=256,
            thinking_config=genai_types.ThinkingConfig(thinking_budget=0),
        ),
    )
    text, error_reason = _safe_extract_text(response)
    if error_reason:
        raise RuntimeError(f"Judge call failed: {error_reason}")
    return _parse_score(text)


# ── Main ──────────────────────────────────────────────────────────────────────

def run_correctness(eval_set_path: str, capture_path: str) -> None:
    eval_rows = load_eval_set(eval_set_path)
    group_a_ids = compute_group_a_ids(eval_rows)      # always 15, fail-loud if not
    captures = load_capture_artifact(capture_path)

    # No eligibility split here — every Group A row is scored, every run.
    # The only reconciliation check is: does every Group A id have a
    # successful capture to judge?
    for rid in group_a_ids:
        cap = captures.get(rid)
        if cap is None or not cap.get("generation_ok"):
            raise RuntimeError(
                f"Group A row {rid} has no successful capture — cannot score. "
                f"Re-run generation_capture.py."
            )

    N = len(group_a_ids)  # == GROUP_A_TOTAL, always

    mlflow.set_experiment(config.MLFLOW_EXPERIMENT_NAME)

    judge_client = get_client()
    judge_model_name = os.getenv("GEMINI_MODEL_NAME", "gemini-2.5-flash")

    scored = load_partial_scores(SCORES_SCRATCH_PATH)
    scored = {rid: r for rid, r in scored.items() if rid in group_a_ids}

    made_live_call = False
    for rid in group_a_ids:
        if rid in scored:
            logger.info(f"Skipped {rid} (already judged).")
            continue

        if made_live_call:
            time.sleep(THROTTLE_SECONDS)

        cap = captures[rid]
        question = eval_rows[rid]["query"]
        gold_answer = eval_rows[rid]["expected_answer_summary"]
        score, reason = judge_correctness(
            judge_client, judge_model_name, question, gold_answer, cap["generated_answer"]
        )
        made_live_call = True
        scored[rid] = {"id": rid, "correctness": score, "reason": reason}
        save_partial_scores(SCORES_SCRATCH_PATH, scored)
        logger.info(f"Judged {rid}: correctness={score}")

    per_row = [scored[rid] for rid in group_a_ids]
    mean_correctness = sum(r["correctness"] for r in per_row) / N

    with mlflow.start_run(run_name="correctness_eval"):
        mlflow.log_metric("correctness_mean", mean_correctness)
        mlflow.log_metric("n_scored", N)
        mlflow.set_tag("gate", "none_scores_all_group_a")  # explicit: no eligibility filter, by design
        mlflow.log_param("eval_set_path", eval_set_path)
        mlflow.log_param("capture_artifact", capture_path)
        mlflow.log_param("judge_model", judge_model_name)
        mlflow.log_dict(
            {
                "correctness_mean": mean_correctness,
                "n_scored": N,
                "group_a_total": GROUP_A_TOTAL,
                "rows": per_row,
            },
            "correctness/per_row.json",
        )
        mlflow.log_artifact(capture_path, artifact_path="correctness_inputs")

    Path(SCORES_SCRATCH_PATH).unlink(missing_ok=True)

    logger.info(f"Correctness mean={mean_correctness:.3f} over N={N} (all Group A, no gate).")


if __name__ == "__main__":
    # ⚠️ Same trap as faithfulness_score.py: this defaults to the arm whose
    # capture file is named below. Scoring a different arm means changing
    # CAPTURE_ARTIFACT to match — there is no recall-run pointer to forget
    # here (no gate), but the capture file itself must still be the right arm.
    CAPTURE_ARTIFACT = "generation_capture_post_frag.json"
    run_correctness(config.EVAL_SET_PATH, CAPTURE_ARTIFACT)


# ── Reading correctness alongside faithfulness ─────────────────────────────
#
# The two metrics are independent axes, not a ranking of "which is more
# right." Read per-row, together:
#
#   High faithfulness + high correctness -> healthy row.
#   High faithfulness + low correctness  -> context didn't support the true
#                                            fact (retrieval gap, or a real
#                                            corpus/eval-set mismatch like q009).
#   Low faithfulness  + high correctness -> model got lucky ungrounded — the
#                                            right answer for the wrong reason.
#   Low faithfulness  + low correctness  -> double failure.
#
# q009 under this scorer: correctness will score it low (a refusal delivers
# none of GOLD's WMI facts) — that's accurate and not a bug. It says nothing
# about whether the refusal was the *right call* given what was retrieved;
# that question belongs to retrieval diagnostics, not either judge.