"""
Build supervised (chat) examples from train.jsonl with optional augmentation.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any, Callable, Iterator

from golf_ft.constraints import ALLOWED_BUILTINS, ALLOWED_MODULES

# Train-only description wraps (identity first).
DESC_AUGMENT_WRAP: tuple[Callable[[str], str], ...] = (
    lambda d: d,
    lambda d: f"Solve in one short lambda: {d}",
    lambda d: f"{d}\n(I/O pairs are JSON below; respond with only a Python lambda.)",
)


def system_prompt() -> str:
    mods = ", ".join(sorted(ALLOWED_MODULES))
    bi = ", ".join(sorted(ALLOWED_BUILTINS - {"__import__"}))
    return (
        "You solve small Python programming tasks by outputting exactly one Python "
        "lambda expression: no statements, no assignments outside the lambda, no markdown, "
        "no explanation. The lambda must work with the given input shapes.\n"
        f"Allowed modules (via __import__ only): {mods}.\n"
        f"Allowed builtins include: {bi}, and __import__ for those modules only.\n"
        "Prefer short, correct code (code golf)."
    )


def serialize_task(
    description: str,
    examples: list[dict[str, Any]],
    difficulty: float | None = None,
) -> str:
    lines: list[str] = [f"Task: {description.strip()}", ""]
    if difficulty is not None:
        lines.append(f"Difficulty (approximate): {difficulty}")
        lines.append("")
    lines.append("Examples (JSON):")
    lines.append(json.dumps(examples, ensure_ascii=False))
    lines.append("\nOutput a single Python lambda only.")
    return "\n".join(lines)


def training_messages(
    description: str,
    examples: list[dict[str, Any]],
    code: str,
    *,
    difficulty: float | None = None,
) -> list[dict[str, str]]:
    user = serialize_task(description, examples, difficulty=difficulty)
    return [
        {"role": "system", "content": system_prompt()},
        {"role": "user", "content": user},
        {"role": "assistant", "content": code.strip()},
    ]


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def augment_examples(
    examples: list[dict[str, Any]],
    rng: random.Random,
    *,
    max_shuffles: int = 3,
) -> Iterator[list[dict[str, Any]]]:
    """
    Yield original plus shuffled-order variants (I/O pairs stay paired).
    Train-only: same semantic label.
    """
    yield list(examples)
    if len(examples) < 2:
        return
    ex = list(examples)
    for _ in range(max_shuffles):
        rng.shuffle(ex)
        yield list(ex)


def augment_descriptions(
    description: str,
    *,
    desc_aug_templates: tuple[Callable[[str], str], ...] | None = None,
) -> Iterator[str]:
    tpl = desc_aug_templates if desc_aug_templates is not None else DESC_AUGMENT_WRAP
    seen: set[str] = set()
    for fn in tpl:
        d = fn(description.strip())
        if d not in seen:
            seen.add(d)
            yield d


def build_train_dataset(
    train_path: Path,
    *,
    seed: int = 42,
    shuffles_per_row: int = 3,
    record_ids: set[int] | None = None,
    use_desc_aug: bool = True,
    desc_aug_templates: tuple[Callable[[str], str], ...] | None = None,
) -> list[list[dict[str, str]]]:
    rng = random.Random(seed)
    records = load_jsonl(train_path)
    out: list[list[dict[str, str]]] = []
    if desc_aug_templates is not None:
        tpl: tuple[Callable[[str], str], ...] = desc_aug_templates
    elif not use_desc_aug:
        tpl = (lambda d: d,)
    else:
        tpl = DESC_AUGMENT_WRAP
    for row in records:
        rid = int(row["id"])
        if record_ids is not None and rid not in record_ids:
            continue
        desc_raw = row["description"]
        code = row["code"]
        diff = row.get("difficulty")
        diff_f = float(diff) if diff is not None else None
        examples: list[dict[str, Any]] = row["examples"]
        for desc in augment_descriptions(desc_raw, desc_aug_templates=tpl):
            for ex_variant in augment_examples(examples, rng, max_shuffles=shuffles_per_row):
                out.append(training_messages(desc, ex_variant, code, difficulty=diff_f))
    return out


def train_val_record_split(
    records: list[dict[str, Any]],
    *,
    val_fraction: float = 0.2,
    seed: int = 42,
) -> tuple[set[int], set[int]]:
    ids = sorted(int(r["id"]) for r in records)
    rng = random.Random(seed)
    rng.shuffle(ids)
    n_val = max(1, int(round(len(ids) * val_fraction)))
    val_ids = set(ids[:n_val])
    train_ids = set(ids[n_val:])
    return train_ids, val_ids


def build_train_val_from_jsonl(
    train_path: Path,
    *,
    val_fraction: float = 0.2,
    seed: int = 42,
    shuffles_per_row: int = 3,
    use_desc_aug: bool = True,
) -> tuple[list[list[dict[str, str]]], list[dict[str, Any]]]:
    records = load_jsonl(train_path)
    train_ids, val_ids = train_val_record_split(records, val_fraction=val_fraction, seed=seed)
    train_msgs = build_train_dataset(
        train_path,
        seed=seed,
        shuffles_per_row=shuffles_per_row,
        record_ids=train_ids,
        use_desc_aug=use_desc_aug,
    )
    val_records = [r for r in records if int(r["id"]) in val_ids]
    return train_msgs, val_records


def save_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def main() -> None:
    ap = argparse.ArgumentParser(description="Build SFT JSON from train.jsonl")
    ap.add_argument(
        "--train",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "train.jsonl",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "data" / "train_messages.json",
    )
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--shuffles", type=int, default=3)
    ap.add_argument(
        "--no-desc-aug",
        action="store_true",
        help="disable description prefix/suffix variants (examples shuffle only)",
    )
    args = ap.parse_args()
    msgs = build_train_dataset(
        args.train,
        seed=args.seed,
        shuffles_per_row=args.shuffles,
        use_desc_aug=not args.no_desc_aug,
    )
    save_json(args.out, msgs)
    print(f"Wrote {len(msgs)} conversations to {args.out}")


if __name__ == "__main__":
    main()
