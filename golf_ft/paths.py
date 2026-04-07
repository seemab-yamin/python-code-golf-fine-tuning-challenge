"""
Canonical locations for challenge assets (organizer files) and generated outputs.

Layout matches the platform zip: released data under ``dataset/public/``;
predictions go under ``working/`` (see README.MD).

Do not duplicate challenge data in ad-hoc paths; import from here or pass CLI overrides.
"""
from __future__ import annotations

from pathlib import Path

REPO_ROOT: Path = Path(__file__).resolve().parent.parent

# Official challenge files from the dataset (train / test / sample submission).
CHALLENGE_DATA_DIR: Path = REPO_ROOT / "dataset" / "public"
TRAIN_JSONL: Path = CHALLENGE_DATA_DIR / "train.jsonl"
TEST_JSONL: Path = CHALLENGE_DATA_DIR / "test.jsonl"
SAMPLE_SUBMISSION_CSV: Path = CHALLENGE_DATA_DIR / "sample_submission.csv"

# Model predictions for upload (platform often expects ./working/submission.csv).
WORKING_DIR: Path = REPO_ROOT / "working"
DEFAULT_SUBMISSION_CSV: Path = WORKING_DIR / "submission.csv"

# Derived training cache (not from organizer; safe to regenerate).
DATA_DIR: Path = REPO_ROOT / "data"
DEFAULT_TRAIN_MESSAGES_JSON: Path = DATA_DIR / "train_messages.json"

# LoRA checkpoints (generated).
DEFAULT_LORA_ADAPTER_DIR: Path = REPO_ROOT / "outputs" / "lora" / "adapter"
