#!/usr/bin/env bash
# Create .venv on Linux and install CUDA-enabled PyTorch, then HF stack.
# Uses virtualenv first (works on Colab / Debian where python3 -m venv breaks on ensurepip).
#
# Usage:
#   ./scripts/setup_venv_linux_gpu.sh
#   export TORCH_CUDA=cu124   # optional: cu121, cu118, cu126 — see https://pytorch.org/get-started/locally/
# Do NOT set TORCH_CUDA to match nvidia-smi "CUDA Version" (e.g. 13.0); use a published wheel tag like cu124.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

TORCH_CUDA="${TORCH_CUDA:-cu124}"
TORCH_INDEX="https://download.pytorch.org/whl/${TORCH_CUDA}"

# Tags PyTorch commonly publishes (not exhaustive); others may still work.
_KNOWN_CUDA_TAGS="cu124 cu121 cu118 cu126 cu128"
if ! echo " ${_KNOWN_CUDA_TAGS} " | grep -q " ${TORCH_CUDA} "; then
  echo "WARNING: TORCH_CUDA=${TORCH_CUDA} is not in the usual set (${_KNOWN_CUDA_TAGS})." >&2
  echo "  nvidia-smi 'CUDA Version' is your driver's max capability, not a pip tag." >&2
  echo "  If install fails, unset TORCH_CUDA or use e.g. export TORCH_CUDA=cu124" >&2
fi

ensure_venv() {
  if [[ -f .venv/bin/activate ]]; then
    return 0
  fi
  if [[ -d .venv ]]; then
    echo "Removing incomplete .venv (no bin/activate)..."
    rm -rf .venv
  fi

  echo "Bootstrapping virtualenv (avoids ensurepip issues on Colab / minimal Python)..."
  python3 -m pip install -q --upgrade pip setuptools wheel
  python3 -m pip install -q "virtualenv>=20.24.0" || python3 -m pip install "virtualenv>=20.24.0"

  if python3 -m virtualenv .venv; then
    echo "Created .venv with: python3 -m virtualenv"
    return 0
  fi

  echo "virtualenv failed; falling back to: python3 -m venv .venv" >&2
  echo "  If this fails with ensurepip, install: sudo apt-get install -y python3-venv" >&2
  python3 -m venv .venv
}

ensure_venv

# shellcheck source=/dev/null
source .venv/bin/activate
python -m pip install --upgrade pip

echo "Installing PyTorch from ${TORCH_INDEX} (TORCH_CUDA=${TORCH_CUDA})"
pip install torch --index-url "${TORCH_INDEX}"

pip install -r requirements-no-torch.txt

echo "Done. Verify GPU:  python -c \"import torch; print(torch.cuda.is_available(), torch.version.cuda)\""
echo "Activate: source .venv/bin/activate"
echo "Optional Hub mirror: source scripts/env_hf_mirror.sh"
