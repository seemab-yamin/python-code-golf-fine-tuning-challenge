"""
LoRA fine-tuning for code-golf lambda generation; held-out train rows for mini-grader eval.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from peft import LoraConfig, get_peft_model
from torch.utils.data import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

from golf_ft.data_pipeline import (
    build_train_val_from_jsonl,
    serialize_task,
    system_prompt,
)
from golf_ft.mini_grader import extract_lambda_string, validate_submission_row


class ChatSFTDataset(Dataset):
    def __init__(
        self,
        tokenizer: Any,
        conversations: list[list[dict[str, str]]],
        max_length: int = 4096,
    ) -> None:
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.rows: list[dict[str, Any]] = []
        for messages in conversations:
            prompt = tokenizer.apply_chat_template(
                messages[:-1], tokenize=False, add_generation_prompt=True
            )
            full = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False
            )
            prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
            full_ids = tokenizer(full, add_special_tokens=False)["input_ids"]
            p_len = len(prompt_ids)
            if len(full_ids) < p_len or full_ids[:p_len] != prompt_ids:
                p_len = 0
            ids = list(full_ids)
            if len(ids) > max_length:
                drop = len(ids) - max_length
                ids = ids[drop:]
                p_len = max(0, p_len - drop)
            labels = [-100] * p_len + ids[p_len:]
            attn = [1] * len(ids)
            self.rows.append(
                {
                    "input_ids": ids,
                    "attention_mask": attn,
                    "labels": labels,
                }
            )

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, i: int) -> dict[str, Any]:
        return self.rows[i]


class Collator:
    def __init__(self, tokenizer: Any) -> None:
        self.pad_id = tokenizer.pad_token_id
        if self.pad_id is None:
            self.pad_id = tokenizer.eos_token_id

    def __call__(self, batch: list[dict[str, Any]]) -> dict[str, Any]:
        max_len = max(len(x["input_ids"]) for x in batch)
        input_ids = []
        attn = []
        labels = []
        for x in batch:
            pad = max_len - len(x["input_ids"])
            input_ids.append(x["input_ids"] + [self.pad_id] * pad)
            attn.append(x["attention_mask"] + [0] * pad)
            labels.append(x["labels"] + [-100] * pad)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attn, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


def _lora_targets(model_name: str) -> list[str]:
    lower = model_name.lower()
    if "qwen" in lower or "llama" in lower or "mistral" in lower or "tinyllama" in lower:
        return [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ]
    return ["c_attn", "c_proj"]  # GPT-2 style


def run_validation(
    model: Any,
    tokenizer: Any,
    val_records: list[dict[str, Any]],
    device: str,
    max_new_tokens: int = 256,
) -> tuple[int, int]:
    model.eval()
    ok_count = 0
    for rec in val_records:
        diff_f = float(rec["difficulty"]) if rec.get("difficulty") is not None else None
        messages = [
            {"role": "system", "content": system_prompt()},
            {
                "role": "user",
                "content": serialize_task(
                    rec["description"], rec["examples"], difficulty=diff_f
                ),
            },
        ]
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            )
        gen = tokenizer.decode(out[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True)
        code = extract_lambda_string(gen)
        good, _ = validate_submission_row(code, rec["examples"], use_subprocess=False)
        if good:
            ok_count += 1
    return ok_count, len(val_records)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--train-jsonl",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "train.jsonl",
    )
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "outputs" / "lora",
    )
    ap.add_argument("--model", type=str, default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--epochs", type=float, default=3.0)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--val-fraction", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-length", type=int, default=4096)
    ap.add_argument(
        "--max-steps",
        type=int,
        default=-1,
        help="if >0, override num_train_epochs and run this many steps (smoke test)",
    )
    ap.add_argument("--skip-train", action="store_true", help="only run val on base model")
    ap.add_argument(
        "--no-desc-aug",
        action="store_true",
        help="disable description variants in training messages",
    )
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    train_msgs, val_records = build_train_val_from_jsonl(
        args.train_jsonl,
        val_fraction=args.val_fraction,
        seed=args.seed,
        shuffles_per_row=3,
        use_desc_aug=not args.no_desc_aug,
    )
    with (args.out_dir / "val_records.json").open("w", encoding="utf-8") as f:
        json.dump(val_records, f, ensure_ascii=False, indent=2)

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    use_cuda = torch.cuda.is_available()
    train_dtype = torch.bfloat16 if use_cuda else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=train_dtype,
        trust_remote_code=True,
    )

    if not args.skip_train:
        lora = LoraConfig(
            r=16,
            lora_alpha=32,
            target_modules=_lora_targets(args.model),
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora)
        ds = ChatSFTDataset(tokenizer, train_msgs, max_length=args.max_length)
        collator = Collator(tokenizer)
        train_kwargs: dict[str, Any] = dict(
            output_dir=str(args.out_dir),
            per_device_train_batch_size=args.batch,
            gradient_accumulation_steps=args.grad_accum,
            learning_rate=args.lr,
            logging_steps=1,
            save_strategy="no",
            bf16=use_cuda,
            fp16=False,
            report_to=[],
        )
        if args.max_steps > 0:
            train_kwargs["max_steps"] = args.max_steps
        else:
            train_kwargs["num_train_epochs"] = args.epochs
        targs = TrainingArguments(**train_kwargs)
        trainer = Trainer(
            model=model,
            args=targs,
            train_dataset=ds,
            data_collator=collator,
        )
        trainer.train()
        model.save_pretrained(args.out_dir / "adapter")
        tokenizer.save_pretrained(args.out_dir / "adapter")

    # Validation pass
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        device = "mps"
    val_dtype = torch.bfloat16 if device == "cuda" else torch.float32
    base = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=val_dtype,
        trust_remote_code=True,
    ).to(device)
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    if (args.out_dir / "adapter" / "adapter_config.json").exists():
        from peft import PeftModel

        base = PeftModel.from_pretrained(base, str(args.out_dir / "adapter"))
    base.eval()
    ok, n = run_validation(base, tok, val_records, device)
    print(f"validation pass (visible examples): {ok}/{n}")
    with (args.out_dir / "val_metrics.json").open("w", encoding="utf-8") as f:
        json.dump({"ok": ok, "n": n}, f)


if __name__ == "__main__":
    main()
