# Linux GPU: setup and full pipeline

Use this on a rented or lab **Linux x86_64 machine with an NVIDIA GPU** (CUDA). You do not need to run training on your laptop.

## 1. Prerequisites

- **NVIDIA driver** installed on the host (e.g. `nvidia-smi` works).
- **Python 3.10+** (3.11 or 3.12 is fine).
- **Git** (for cloning this repository).
- Network access to download models (or use the Hub mirror in §3).

Check GPU:

```bash
nvidia-smi
python3 --version
```

## 2. Get the code and data

On the GPU machine, clone the repository. **Prefer HTTPS** for a public repo (no SSH host-key setup):

```bash
git clone https://github.com/seemab-yamin/python-code-golf-fine-tuning-challenge.git
cd python-code-golf-fine-tuning-challenge
```

SSH (only if you use SSH keys with GitHub and `github.com` is in `~/.ssh/known_hosts`):

```bash
git clone git@github.com:seemab-yamin/python-code-golf-fine-tuning-challenge.git
cd python-code-golf-fine-tuning-challenge
```

**`Host key verification failed`** does *not* mean the repo is private. It means SSH does not trust `github.com` yet (or the session cannot write `known_hosts`). Fix: use **HTTPS** above, or run `ssh-keyscan -t ed25519 github.com >> ~/.ssh/known_hosts` once (compare fingerprints with [GitHub’s SSH docs](https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/githubs-ssh-key-fingerprints) if you need strict verification).

The repo includes challenge assets under **`dataset/public/`**: `train.jsonl`, `test.jsonl`, and `sample_submission.csv`. If that folder is missing files (e.g. LFS or a slim clone), copy them from the official competition zip into `dataset/public/` so paths match `golf_ft/paths.py`.

## 3. Virtual environment and CUDA PyTorch

PyTorch must be installed from the **PyTorch CUDA wheel index**; a plain `pip install torch` often gives a CPU-only build.

### Colab-friendly venv (`virtualenv`, not `python3 -m venv`)

On **Google Colab** and many **Debian** images, `python3 -m venv` fails with **`ensurepip` errors**. [`scripts/setup_venv_linux_gpu.sh`](scripts/setup_venv_linux_gpu.sh) creates **`.venv` with `virtualenv`** first, then falls back to `python3 -m venv` only if needed.

**Shell tip:** do not type `!chmod` in bash — `!` triggers history expansion. Use `chmod +x ...` only.

**`TORCH_CUDA` vs nvidia-smi:** the line **CUDA Version: 13.0** in `nvidia-smi` is the **driver’s maximum** capability, **not** a PyTorch wheel name. There is typically **no** `cu130` index. Use a published tag (default **`cu124`**) unless [PyTorch’s install matrix](https://pytorch.org/get-started/locally/) shows otherwise:

```bash
cd ~/python-code-golf-fine-tuning-challenge
chmod +x scripts/setup_venv_linux_gpu.sh
# Default TORCH_CUDA=cu124 is usually correct for a T4 + recent driver
./scripts/setup_venv_linux_gpu.sh
source .venv/bin/activate
```

Optional: another wheel bundle (if listed on PyTorch’s site):

```bash
export TORCH_CUDA=cu121
./scripts/setup_venv_linux_gpu.sh
```

The script **warns** if `TORCH_CUDA` is not a common tag but still tries the URL.

**Manual virtualenv** (if you skip the script’s create step):

```bash
python3 -m pip install virtualenv
rm -rf .venv   # if a broken partial venv exists
python3 -m virtualenv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install torch --index-url "https://download.pytorch.org/whl/cu124"
pip install -r requirements-no-torch.txt
```

**Alternates**

- **Apt + stdlib venv** (VM with sudo): `sudo apt-get install -y python3.12-venv` then `python3 -m venv .venv` and install torch as above.
- **Colab, no venv:** if the runtime already has `torch` + CUDA, run `pip install transformers peft accelerate` only and use the repo from `/content/...`.

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

By default this writes **`working/submission.csv`** (see `golf_ft/paths.py`). Run `submission_qa` on that path if you skipped QA inside `infer`.

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
path_to_test="$(pwd)/dataset/public/test.jsonl"
python -m golf_ft.submission_qa --csv working/submission.csv --test "$path_to_test" --replay-visible
```

## 9. Optional: train-set weighted score (tuning)

If you generated predictions for **train** task ids with reference code in `dataset/public/train.jsonl`:

```bash
python -m golf_ft.score_train --csv my_train_predictions.csv
```

## 10. Copy results back to your machine

```bash
# From your laptop
scp user@your-gpu-host:~/python-code-golf-fine-tuning-challenge/working/submission.csv .
scp -r user@your-gpu-host:~/python-code-golf-fine-tuning-challenge/outputs/lora/adapter ./adapter-backup
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
4. `python -m golf_ft.data_pipeline` (reads `dataset/public/train.jsonl`).
5. `python -m golf_ft.train_lora ...` (or `--max-steps 1` smoke).
6. `mkdir -p working` → `python -m golf_ft.infer --out working/submission.csv --device cuda`.
7. `python -m golf_ft.submission_qa --csv working/submission.csv`.
8. Download `working/submission.csv` (and your `solution.py` / notebook) for the challenge platform.
