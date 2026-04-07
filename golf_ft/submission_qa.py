"""
Validate submission.csv against README requirements; optional visible-example replay.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

from golf_ft.data_pipeline import load_jsonl
from golf_ft.mini_grader import validate_submission_row


def validate_submission_csv(
    csv_path: Path,
    *,
    test_jsonl: Path | None = None,
    replay_visible: bool = False,
    use_subprocess: bool = True,
    timeout_s: float = 5.0,
) -> tuple[bool, list[str]]:
    """
    Check header, exactly 200 rows, ids 1..200 each once, non-empty codes.
    If test_jsonl and replay_visible, run mini_grader on released examples per id.
    """
    issues: list[str] = []
    expected = set(range(1, 201))

    with csv_path.open(newline="", encoding="utf-8") as f:
        r = csv.reader(f)
        rows_list = list(r)

    if not rows_list:
        return False, ["empty CSV"]

    header = rows_list[0]
    if [h.strip() for h in header] != ["id", "code"]:
        issues.append(f'bad header: {header!r} (want ["id", "code"])')

    data_rows = rows_list[1:]
    if len(data_rows) != 200:
        issues.append(f"expected 200 data rows, got {len(data_rows)}")

    seen: set[int] = set()
    codes_by_id: dict[int, str] = {}

    for i, row in enumerate(data_rows, start=2):
        if len(row) < 2:
            issues.append(f"line {i}: not enough columns")
            continue
        try:
            rid = int(row[0].strip())
        except ValueError:
            issues.append(f"line {i}: bad id {row[0]!r}")
            continue
        if rid in seen:
            issues.append(f"duplicate id {rid}")
        seen.add(rid)
        code = row[1].strip() if len(row) > 1 else ""
        if not code:
            issues.append(f"id {rid}: empty code")
        codes_by_id[rid] = row[1] if len(row) > 1 else ""

    missing = expected - seen
    extra = seen - expected
    if missing:
        issues.append(f"missing ids: {sorted(missing)[:20]}{'...' if len(missing) > 20 else ''}")
    if extra:
        issues.append(f"unexpected ids: {sorted(extra)}")

    examples_by_id: dict[int, list[dict[str, Any]]] | None = None
    if test_jsonl is not None and replay_visible:
        examples_by_id = {}
        for rec in load_jsonl(test_jsonl):
            examples_by_id[int(rec["id"])] = rec["examples"]

        for rid in sorted(expected):
            code = codes_by_id.get(rid, "").strip()
            if not code or rid not in examples_by_id:
                continue
            ok, errs = validate_submission_row(
                code,
                examples_by_id[rid],
                use_subprocess=use_subprocess,
                timeout_s=timeout_s,
            )
            if not ok:
                issues.append(f"id {rid} visible replay failed: {'; '.join(errs)}")

    ok = len(issues) == 0
    return ok, issues


def main() -> None:
    ap = argparse.ArgumentParser(description="Validate submission.csv")
    ap.add_argument(
        "--csv",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "submission.csv",
    )
    ap.add_argument(
        "--test",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "test.jsonl",
    )
    ap.add_argument(
        "--replay-visible",
        action="store_true",
        help="run mini-grader on visible examples from test.jsonl",
    )
    ap.add_argument(
        "--no-subprocess",
        action="store_true",
    )
    ap.add_argument(
        "--timeout",
        type=float,
        default=5.0,
        help="per-example timeout when using subprocess replay",
    )
    args = ap.parse_args()

    ok, issues = validate_submission_csv(
        args.csv,
        test_jsonl=args.test if args.replay_visible else None,
        replay_visible=args.replay_visible,
        use_subprocess=not args.no_subprocess,
        timeout_s=args.timeout,
    )
    if ok:
        print("OK:", args.csv)
    else:
        print("FAILED:", args.csv)
        for msg in issues:
            print(" ", msg)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
