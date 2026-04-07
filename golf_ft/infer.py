"""
Generate submission.csv from test.jsonl using a local base model + optional LoRA adapter.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from golf_ft.data_pipeline import load_jsonl, serialize_task, system_prompt
from golf_ft.mini_grader import extract_lambda_string, validate_submission_row
from golf_ft.paths import (
    DEFAULT_LORA_ADAPTER_DIR,
    DEFAULT_SUBMISSION_CSV,
    TEST_JSONL,
)
from golf_ft.submission_qa import validate_submission_csv

# Best-of-N generations; pick shortest correct on visible examples (README rewards brevity).
DEFAULT_INFER_ATTEMPTS: list[tuple[float, int]] = [
    (0.0, 256),
    (0.0, 384),
    (0.12, 320),
    (0.25, 448),
    (0.35, 512),
]
QUICK_INFER_ATTEMPTS: list[tuple[float, int]] = [(0.0, 256), (0.3, 384)]


def load_model_and_tokenizer(
    model_name: str,
    adapter_dir: Path | None,
    device: str,
) -> tuple[Any, Any]:
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    m = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        trust_remote_code=True,
    ).to(device)
    if adapter_dir and (adapter_dir / "adapter_config.json").exists():
        from peft import PeftModel

        m = PeftModel.from_pretrained(m, str(adapter_dir))
    m.eval()
    return m, tok


def generate_lambda(
    model: Any,
    tokenizer: Any,
    description: str,
    examples: list[dict[str, Any]],
    device: str,
    max_new_tokens: int = 256,
    temperature: float = 0.0,
    difficulty: float | None = None,
) -> str:
    messages = [
        {"role": "system", "content": system_prompt()},
        {"role": "user", "content": serialize_task(description, examples, difficulty=difficulty)},
    ]
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    gen_kw: dict[str, Any] = {
        "max_new_tokens": max_new_tokens,
        "pad_token_id": tokenizer.pad_token_id or tokenizer.eos_token_id,
    }
    if temperature and temperature > 0:
        gen_kw["do_sample"] = True
        gen_kw["temperature"] = temperature
    else:
        gen_kw["do_sample"] = False
    with torch.no_grad():
        out = model.generate(**inputs, **gen_kw)
    text = tokenizer.decode(
        out[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True
    )
    return extract_lambda_string(text)


def pick_best_code(
    model: Any,
    tokenizer: Any,
    description: str,
    examples: list[dict[str, Any]],
    device: str,
    attempts: list[tuple[float, int]],
    *,
    use_subprocess: bool = True,
    difficulty: float | None = None,
    timeout_s: float = 5.0,
) -> str:
    """Try (temperature, max_tokens) pairs; prefer shortest valid on visible examples."""
    candidates: list[str] = []
    for temp, mnt in attempts:
        raw = generate_lambda(
            model,
            tokenizer,
            description,
            examples,
            device,
            max_new_tokens=mnt,
            temperature=temp,
            difficulty=difficulty,
        )
        ok, _ = validate_submission_row(
            raw,
            examples,
            use_subprocess=use_subprocess,
            timeout_s=timeout_s,
        )
        if ok:
            candidates.append(raw.strip())
    if candidates:
        return min(candidates, key=len)
    # fallback: parseable minimal identity (may score 0 on grader)
    return "lambda x:x"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--test-jsonl",
        type=Path,
        default=TEST_JSONL,
        help="default: dataset/public/test.jsonl (challenge file)",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_SUBMISSION_CSV,
        help="default: working/submission.csv",
    )
    ap.add_argument("--model", type=str, default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument(
        "--adapter",
        type=Path,
        default=DEFAULT_LORA_ADAPTER_DIR,
    )
    ap.add_argument("--no-adapter", action="store_true")
    ap.add_argument(
        "--no-subprocess",
        action="store_true",
        help="run visible examples in-process (faster; no hard timeout)",
    )
    ap.add_argument(
        "--timeout",
        type=float,
        default=5.0,
        help="per-row timeout for visible-example checks (seconds)",
    )
    ap.add_argument(
        "--skip-qa",
        action="store_true",
        help="do not validate submission.csv shape after write",
    )
    ap.add_argument(
        "--qa-replay-visible",
        action="store_true",
        help="during QA, re-run mini-grader on visible test examples",
    )
    ap.add_argument("--device", type=str, default="auto")
    ap.add_argument(
        "--quick-infer",
        action="store_true",
        help="only two (temperature, max_new_tokens) attempts (faster; less breadth for short lambdas)",
    )
    ap.add_argument(
        "--model-row-limit",
        type=int,
        default=-1,
        help="if >= 0, only first N rows (in test.jsonl order) use the model; others get lambda x:x. Use for smoke tests; full submit should use -1 (default).",
    )
    args = ap.parse_args()

    if args.device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    else:
        device = args.device

    adapter = None if args.no_adapter else args.adapter
    model, tokenizer = load_model_and_tokenizer(args.model, adapter, device)

    rows = load_jsonl(args.test_jsonl)
    attempts = QUICK_INFER_ATTEMPTS if args.quick_infer else DEFAULT_INFER_ATTEMPTS
    limit = args.model_row_limit

    out_rows: list[tuple[int, str]] = []
    for idx, rec in enumerate(rows):
        rid = int(rec["id"])
        diff_f = float(rec["difficulty"]) if rec.get("difficulty") is not None else None
        if limit >= 0 and idx >= limit:
            code = "lambda x:x"
        else:
            code = pick_best_code(
                model,
                tokenizer,
                rec["description"],
                rec["examples"],
                device,
                attempts,
                use_subprocess=not args.no_subprocess,
                difficulty=diff_f,
                timeout_s=args.timeout,
            )
        out_rows.append((rid, code))
        print(f"id {rid} -> {code[:80]!r}{'...' if len(code) > 80 else ''}")

    out_rows.sort(key=lambda x: x[0])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "code"])
        for rid, code in out_rows:
            if not code.strip():
                code = "lambda x:x"
            w.writerow([rid, code])

    print(f"Wrote {len(out_rows)} rows to {args.out}")

    if not args.skip_qa:
        ok, issues = validate_submission_csv(
            args.out,
            test_jsonl=args.test_jsonl if args.qa_replay_visible else None,
            replay_visible=args.qa_replay_visible,
            use_subprocess=not args.no_subprocess,
            timeout_s=args.timeout,
        )
        if not ok:
            print("submission QA failed:")
            for msg in issues:
                print(" ", msg)
            raise SystemExit(1)
        print("submission QA passed:", args.out)


if __name__ == "__main__":
    main()
