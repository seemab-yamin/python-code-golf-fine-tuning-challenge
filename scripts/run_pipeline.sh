#!/usr/bin/env bash
# End-to-end: data -> LoRA train -> infer -> submission QA (README gap plan step 7).
# Usage:
#   ./scripts/run_pipeline.sh              full train + full infer (slow)
#   SMOKE=1 ./scripts/run_pipeline.sh      tiny train + 5 model rows + shape QA
# Optional: NO_MIRROR=1 to skip hf-mirror.com (default sources scripts/env_hf_mirror.sh).
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if [[ "${NO_MIRROR:-}" != "1" ]] && [[ -f scripts/env_hf_mirror.sh ]]; then
  # shellcheck source=/dev/null
  source scripts/env_hf_mirror.sh
fi

if [[ ! -x .venv/bin/python ]]; then
  echo "Run ./scripts/setup_venv.sh first"
  exit 1
fi
# shellcheck source=/dev/null
source .venv/bin/activate

python -m golf_ft.data_pipeline

if [[ "${SMOKE:-}" == "1" ]]; then
  echo "[SMOKE] train_lora --max-steps 2"
  python -m golf_ft.train_lora --max-steps 2 --batch 1 --grad-accum 1 --val-fraction 0.2
  echo "[SMOKE] infer (first 5 rows model; rest placeholder), quick, in-process checks"
  python -m golf_ft.infer \
    --quick-infer \
    --no-subprocess \
    --model-row-limit 5 \
    --skip-qa
  python -m golf_ft.submission_qa --csv submission.csv --no-subprocess
else
  python -m golf_ft.train_lora
  python -m golf_ft.infer
fi

echo "Done. Optional: python -m golf_ft.submission_qa --csv submission.csv --replay-visible --test test.jsonl"
