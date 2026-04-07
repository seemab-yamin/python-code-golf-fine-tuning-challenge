#!/usr/bin/env bash
# Source this before training/infer if huggingface.co is unreachable:
#   source scripts/env_hf_mirror.sh
# Uses the public Hub API mirror (see https://hf-mirror.com).
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
