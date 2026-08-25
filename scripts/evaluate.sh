#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-python}"
MODEL_NAME="${MODEL_NAME:-Qwen/Qwen2.5-VL-7B-Instruct}"
CHECKPOINT="${CHECKPOINT:-outputs/stage3/best}"
TEST_JSONL="${TEST_JSONL:-data/vgthink/test.jsonl}"
IMAGE_ROOT="${IMAGE_ROOT:-data}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/eval}"
DEVICE="${DEVICE:-cuda:0}"

"${PYTHON_BIN}" evaluate.py \
  --model_name "${MODEL_NAME}" \
  --checkpoint "${CHECKPOINT}" \
  --test_jsonl "${TEST_JSONL}" \
  --image_root "${IMAGE_ROOT}" \
  --output_dir "${OUTPUT_DIR}" \
  --system_prompt_file configs/system_prompt.txt \
  --device "${DEVICE}" \
  --bf16
