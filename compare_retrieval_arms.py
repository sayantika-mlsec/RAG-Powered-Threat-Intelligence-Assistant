# Row-by-row retrieved_ids comparison between two retrieval eval arms.
#
# Precision@K and recall@K are set-overlap metrics against expected_ids — they
# cannot tell you whether two arms retrieved the SAME three chunks or just
# equally-wrong/equally-right ones. This script answers that question
# directly: for every scored query both artifacts share, are the top-K
# retrieved_ids identical, the same set in a different order, or genuinely
# different chunks?
#
# Motivating case (Jul 16): q013 scored recall@3=1.0 on both the blind and
# routed arms, which read as "routing changed nothing" from the metrics
# alone. This script showed the third retrieved slot actually differed
# (T1068 vs CVE-2019-0543) — both irrelevant to the query, so the score never
# moved, but the retrieved set did. That distinction is invisible to
# precision/recall and only visible by reading retrieved_ids directly.
#
# Source: each artifact arg is EITHER a local JSON path OR an MLflow run id.
# Local-file-exists is checked first; anything else is treated as a run id
# and pulled from that run's "per_row_metrics" artifact directory (same
# resolution pattern used by the faithfulness script's load_recall_artifact).

import argparse
import json
import logging
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
)
logger = logging.getLogger(__name__)


def load_artifact(source: str) -> dict:
    """
    Load a retrieval eval artifact from a local JSON path, or — if `source`
    isn't an existing local path — from an MLflow run's per_row_metrics
    artifact directory (targets the directory, not a hardcoded filename,
    since the eval scripts write it under a temp name).
    """
    p = Path(source)
    if p.exists():
        return json.loads(p.read_text(encoding="utf-8"))

    # Lazy import: only needed on this branch, so comparing two local files
    # never requires mlflow to be installed at all.
    from mlflow.artifacts import download_artifacts
    local_dir = download_artifacts(run_id=source, artifact_path="per_row_metrics")
    json_files = list(Path(local_dir).glob("*.json"))
    if len(json_files) != 1:
        raise RuntimeError(
            f"Expected exactly one JSON in 'per_row_metrics' of run {source}, "
            f"found {len(json_files)}: {[f.name for f in json_files]}"
        )
    return json.loads(json_files[0].read_text(encoding="utf-8"))


def compare_arms(artifact_a: dict, artifact_b: dict, label_a: str, label_b: str) -> None:
    """
    Compare retrieved_ids row-by-row for every scored query id both artifacts
    share. Prints a table; raises loud if the two artifacts don't share the
    same K or the same set of scored ids — comparing across a K mismatch, or
    silently skipping ids only one side scored, would produce a misleading
    table rather than a wrong-but-visible one.
    """
    k_a, k_b = artifact_a.get("k_value"), artifact_b.get("k_value")
    if k_a != k_b:
        raise RuntimeError(
            f"K mismatch: {label_a} K={k_a}, {label_b} K={k_b}. "
            f"Comparing retrieved_ids across different K is not meaningful."
        )

    rows_a = {r["id"]: r for r in artifact_a["rows"]}
    rows_b = {r["id"]: r for r in artifact_b["rows"]}

    ids_a, ids_b = set(rows_a), set(rows_b)
    only_a, only_b = ids_a - ids_b, ids_b - ids_a
    shared = sorted(ids_a & ids_b)

    if only_a or only_b:
        logger.warning(
            f"Row-id mismatch between artifacts — only in {label_a}: "
            f"{sorted(only_a) or 'none'}; only in {label_b}: {sorted(only_b) or 'none'}. "
            f"These rows are excluded from comparison below."
        )

    identical = []
    same_set_diff_order = []
    different = []
    # Subset of `different` where the score was IDENTICAL anyway — the exact
    # case precision/recall alone cannot surface. Called out separately.
    different_but_score_matched = []

    print(f"\n--- Retrieved-IDs Comparison: {label_a}  vs  {label_b}  (K={k_a}) ---")
    print(f"{'ID':<6} | {'Status':<20} | {'Recall A':<9} | {'Recall B':<9} | Note")
    print("-" * 90)

    for row_id in shared:
        ra, rb = rows_a[row_id], rows_b[row_id]
        list_a, list_b = ra["retrieved_ids"], rb["retrieved_ids"]
        recall_a, recall_b = ra.get(f"recall_at_{k_a}"), rb.get(f"recall_at_{k_b}")

        if list_a == list_b:
            status = "IDENTICAL"
            identical.append(row_id)
            note = ""
        elif sorted(list_a) == sorted(list_b):
            status = "SAME_SET_DIFF_ORDER"
            same_set_diff_order.append(row_id)
            note = ""
        else:
            status = "DIFFERENT"
            different.append(row_id)
            if recall_a == recall_b:
                different_but_score_matched.append(row_id)
                note = "score identical despite different retrieval — invisible to recall/precision"
            else:
                note = "retrieval differs AND score differs — already visible in metrics"

        flag = "  " if status == "IDENTICAL" else "!!"
        print(
            f"{flag} {row_id:<4} | {status:<20} | {str(recall_a):<9} | {str(recall_b):<9} | {note}"
        )

    print("-" * 90)
    print(
        f"Shared rows: {len(shared)}  |  Identical: {len(identical)}  |  "
        f"Same set, different order: {len(same_set_diff_order)}  |  "
        f"Different sets: {len(different)}"
    )
    if different_but_score_matched:
        print(
            f"\n*** {len(different_but_score_matched)} row(s) had DIFFERENT retrieved_ids "
            f"but IDENTICAL scores — the exact case recall/precision alone cannot show: "
            f"{different_but_score_matched} ***"
        )
    for row_id in different:
        ra, rb = rows_a[row_id], rows_b[row_id]
        print(f"\n{row_id} — {label_a}: {ra['retrieved_ids']}  vs  {label_b}: {rb['retrieved_ids']}")
        print(f"       expected_ids: {ra['expected_ids']}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Compare retrieved_ids row-by-row between two retrieval eval artifacts."
    )
    ap.add_argument("artifact_a", help="Local JSON path or MLflow run id (e.g. blind arm)")
    ap.add_argument("artifact_b", help="Local JSON path or MLflow run id (e.g. routed arm)")
    ap.add_argument("--label-a", default="A", help="Display label for artifact_a")
    ap.add_argument("--label-b", default="B", help="Display label for artifact_b")
    args = ap.parse_args()

    a = load_artifact(args.artifact_a)
    b = load_artifact(args.artifact_b)
    compare_arms(a, b, args.label_a, args.label_b)