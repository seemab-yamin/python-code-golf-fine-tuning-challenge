# Linux GPU: setup and full pipeline

Use this on a rented or lab **Linux x86_64 machine with an NVIDIA GPU** (CUDA). You do not need to run training on your laptop.

## 1. Prerequisites

- **NVIDIA driver** installed on the host (e.g. `nvidia-smi` works).
- **Python 3.10+** (3.11 or 3.12 is fine).
- **Git** (optional) or copy the project directory to the server as a zip.
- Network access to download models (or use the Hub mirror in §3).

Check GPU:

```bash
nvidia-smi
python3 --version
```

## 2. Get the code and data

From your laptop, copy the whole challenge folder to the server (example with `scp`):

```bash
scp -r "/path/to/Python Code Golf Fine-Tuning Challenge" user@your-gpu-host:~/code-golf-ft
ssh user@your-gpu-host
cd ~/code-golf-ft
```

Ensure these files exist at the repo root: `train.jsonl`, `test.jsonl`, `sample_submission.csv`, and the `golf_ft/` package.

## 3. Virtual environment and CUDA PyTorch

PyTorch must be installed from the **PyTorch CUDA wheel index**; a plain `pip install torch` often gives a CPU-only build.

Pick a CUDA build that matches your driver (see [PyTorch install](https://pytorch.org/get-started/locally/)). Common choice today is **cu124**:

```bash
cd ~/code-golf-ft
chmod +x scripts/setup_venv_linux_gpu.sh
```

Optional: pick another bundle (examples: `cu121`, `cu118`):

```bash
export TORCH_CUDA=cu124
./scripts/setup_venv_linux_gpu.sh
source .venv/bin/activate
```

Verify CUDA from Python:

```bash
python -c "import torch; print('cuda:', torch.cuda.is_available(), 'device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)"
```

### If huggingface.co is blocked or slow

```bash
source scripts/env_hf_mirror.sh
```

This sets `HF_ENDPOINT=https://hf-mirror.com` for `transformers` / `huggingface_hub`.

## 4. One-shot pipeline script (optional)

If the venv already has a working GPU `torch` (§3), you can run data → train → infer in one go:

```bash
source .venv/bin/activate
# Optional: SMOKE=1 for a short train (--max-steps 2) and partial infer (see script).
NO_MIRROR=1 ./scripts/run_pipeline.sh          # use huggingface.co
# or omit NO_MIRROR to source scripts/env_hf_mirror.sh by default
./scripts/run_pipeline.sh
```

By default this writes `submission.csv` in the **repo root** (not under `working/`). After it finishes, move or rerun infer with `--out working/submission.csv` if the platform expects that path, and run `submission_qa` on the same path.

## 5. Build training conversations (optional cache)

Regenerates `data/train_messages.json` (takes seconds):

```bash
python -m golf_ft.data_pipeline
```

## 6. Fine-tune (LoRA)

Default model is `Qwen/Qwen2.5-0.5B-Instruct` (small; adjust `--model` if you want a larger code model). Adapter and logs go under `outputs/lora/`.

Example full training pass:

```bash
python -m golf_ft.train_lora \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --out-dir outputs/lora \
  --epochs 3 \
  --batch 1 \
  --grad-accum 8 \
  --lr 2e-4
```

Smoke test (one optimization step only):

```bash
python -m golf_ft.train_lora --out-dir outputs/lora_smoke --max-steps 1 --batch 1 --grad-accum 1
```

- Training uses **bfloat16** when `torch.cuda.is_available()` (see `train_lora.py`).
- Validation on a held-out slice of **train** rows is printed at the end; it runs visible examples only (same as local mini-grader).

To skip training and only run validation on the base model:

```bash
python -m golf_ft.train_lora --skip-train --out-dir outputs/lora
```

## 7. Inference and submission CSV

Create the platform layout directory and write predictions where many graders expect them:

```bash
mkdir -p working
python -m golf_ft.infer \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --adapter outputs/lora/adapter \
  --out working/submission.csv \
  --device cuda
```

If you did **not** train an adapter (or saved elsewhere):

```bash
python -m golf_ft.infer --no-adapter --out working/submission.csv --device cuda
```

Flags you may need:

- `--no-subprocess` — faster visible checks in-process on Linux (no per-row subprocess); optional.
- `--skip-qa` — disable post-write CSV shape checks (not recommended).
- `--qa-replay-visible` — after writing, re-run mini-grader on all **visible** `test.jsonl` examples (slow but strict).

## 8. Validate the CSV without re-running inference

Shape only (200 rows, ids 1–200, header, non-empty codes):

```bash
python -m golf_ft.submission_qa --csv working/submission.csv
```

Include visible-example replay:

```bash
path_to_test="$(pwd)/test.jsonl"
python -m golf_ft.submission_qa --csv working/submission.csv --test "$path_to_test" --replay-visible
```

## 9. Optional: train-set weighted score (tuning)

If you generated predictions for **train** task ids with reference code in `train.jsonl`:

```bash
python -m golf_ft.score_train --csv my_train_predictions.csv
```

## 10. Copy results back to your machine

```bash
# From your laptop
scp user@your-gpu-host:~/code-golf-ft/working/submission.csv .
scp -r user@your-gpu-host:~/code-golf-ft/outputs/lora/adapter ./adapter-backup
```

## 11. Troubleshooting

| Issue | What to try |
|--------|-------------|
| `torch.cuda.is_available()` is False | Reinstall torch with the correct `--index-url` / `TORCH_CUDA`; confirm `nvidia-smi`. |
| OOM during training | Lower `--batch`, increase `--grad-accum`, use a smaller `--model`, or shorten `--max-length` in code. |
| Hub timeout | `source scripts/env_hf_mirror.sh` or retry; check firewall. |
| `infer` is slow | Normal on CPU; always pass `--device cuda` on the GPU box. |
| QA fails visible replay | Model quality issue; iterate on epochs, model size, or prompts—not a CSV format bug. |

## End-to-end checklist

1. `nvidia-smi` works.
2. `./scripts/setup_venv_linux_gpu.sh` and `torch.cuda.is_available()` is True.
3. (Optional) `source scripts/env_hf_mirror.sh`.
4. `python -m golf_ft.data_pipeline`.
5. `python -m golf_ft.train_lora ...` (or `--max-steps 1` smoke).
6. `mkdir -p working` → `python -m golf_ft.infer --out working/submission.csv --device cuda`.
7. `python -m golf_ft.submission_qa --csv working/submission.csv`.
8. Download `working/submission.csv` (and your `solution.py` / notebook) for the challenge platform.
