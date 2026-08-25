#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-python}"
MODEL_NAME="${MODEL_NAME:-Qwen/Qwen2.5-VL-7B-Instruct}"
ADAPTER_PATH="${ADAPTER_PATH:-outputs/stage2/best}"
TRAIN_JSONL="${TRAIN_JSONL:-data/vgthink/train.jsonl}"
DEV_JSONL="${DEV_JSONL:-data/vgthink/dev.jsonl}"
IMAGE_ROOT="${IMAGE_ROOT:-data}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/stage3}"
DEVICE="${DEVICE:-cuda:0}"

"${PYTHON_BIN}" train_grpo.py \
  --model_name "${MODEL_NAME}" \
  --adapter_path "${ADAPTER_PATH}" \
  --train_jsonl "${TRAIN_JSONL}" \
  --dev_jsonl "${DEV_JSONL}" \
  --image_root "${IMAGE_ROOT}" \
  --output_dir "${OUTPUT_DIR}" \
  --system_prompt_file configs/system_prompt.txt \
  --device "${DEVICE}" \
  --epochs 1 \
  --num_rollouts 4 \
  --bf16
