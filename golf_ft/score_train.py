"""
Local README-style difficulty-weighted score using train.jsonl reference `code`.
Only rows with correct visible examples contribute; incorrect rows contribute 0.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from golf_ft.data_pipeline import load_jsonl
from golf_ft.mini_grader import validate_submission_row


def train_weighted_golf_score(
    csv_path: Path,
    train_jsonl: Path,
    *,
    use_subprocess: bool = False,
    timeout_s: float = 5.0,
) -> tuple[float, int, int, list[str]]:
    """
    score = 100 * sum(d_i * golf_i) / sum(d_i) with golf_i = len(ref)/len(sub) if correct else 0.
    Returns (score, n_correct, n_matched_train_rows, n_skipped_rows, notes).
    Use a CSV whose ids are train task ids (e.g. predictions for train.jsonl), not a 200-row test submission.
    """
    train = load_jsonl(train_jsonl)
    by_id = {int(r["id"]): r for r in train}

    with csv_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    num = 0.0
    den = 0.0
    correct = 0
    notes: list[str] = []
    skipped = 0

    for row in rows:
        rid = int(row["id"])
        if rid not in by_id:
            skipped += 1
            notes.append(f"id {rid}: not in train.jsonl (skipped)")
            continue
        rec = by_id[rid]
        diff = float(rec["difficulty"])
        den += diff
        ref = rec["code"].strip()
        sub = (row.get("code") or "").strip()
        ok, _ = validate_submission_row(
            sub,
            rec["examples"],
            use_subprocess=use_subprocess,
            timeout_s=timeout_s,
        )
        if ok and sub:
            golf = len(ref) / len(sub)
            num += diff * golf
            correct += 1
        else:
            notes.append(f"id {rid}: incorrect or empty (0 golf contribution)")

    score = 100.0 * num / den if den > 0 else 0.0
    n_matched = len(rows) - skipped
    return score, correct, n_matched, skipped, notes


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Score a CSV against train.jsonl references (use train task ids, not test submission.csv)"
    )
    ap.add_argument(
        "--csv",
        type=Path,
        required=True,
    )
    ap.add_argument(
        "--train",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "train.jsonl",
    )
    ap.add_argument("--subprocess", action="store_true")
    args = ap.parse_args()

    score, ok, n_matched, skipped, notes = train_weighted_golf_score(
        args.csv,
        args.train,
        use_subprocess=args.subprocess,
    )
    print(f"train-weighted score (README formula, visible-only check): {score:.4f}")
    print(f"correct on visible examples (train ids only): {ok}/{n_matched}")
    if skipped:
        print(
            f"note: skipped {skipped} CSV rows with id not in train.jsonl "
            f"(full test submission.csv is not meant for this tool)"
        )
    for line in notes[:30]:
        print(line)
    if len(notes) > 30:
        print(f"... and {len(notes) - 30} more")


if __name__ == "__main__":
    main()
