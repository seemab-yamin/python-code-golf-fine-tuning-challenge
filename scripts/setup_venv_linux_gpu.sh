#!/usr/bin/env bash
# Create .venv on Linux and install CUDA-enabled PyTorch, then HF stack.
# Usage:
#   export TORCH_CUDA=cu124   # optional: cu121, cu118 — match https://pytorch.org/get-started/locally/
#   ./scripts/setup_venv_linux_gpu.sh
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

TORCH_CUDA="${TORCH_CUDA:-cu124}"
TORCH_INDEX="https://download.pytorch.org/whl/${TORCH_CUDA}"

if [[ ! -d .venv ]]; then
  python3 -m venv .venv
fi
# shellcheck source=/dev/null
source .venv/bin/activate
python -m pip install --upgrade pip

echo "Installing PyTorch from ${TORCH_INDEX} (TORCH_CUDA=${TORCH_CUDA})"
pip install torch --index-url "${TORCH_INDEX}"

pip install -r requirements-no-torch.txt

echo "Done. Verify GPU:  python -c \"import torch; print(torch.cuda.is_available(), torch.version.cuda)\""
echo "Activate: source .venv/bin/activate"
echo "Optional Hub mirror: source scripts/env_hf_mirror.sh"
