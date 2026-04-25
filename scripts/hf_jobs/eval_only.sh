#!/usr/bin/env bash
# HF Jobs entrypoint: evaluate any pre-trained adapter against the two-tier
# held-out under the hardened rubric. Cheap (no GPU training) — useful for
# reproducing reported numbers on a sponsor-provided GPU.
set -euo pipefail

echo "=== HF Job: two-tier eval only (no training) ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

echo "[1/4] apt deps"
apt-get update -qq
apt-get install -y -qq git curl

echo "[2/4] git clone repo @ HEAD"
cd /tmp
git clone --depth=1 https://github.com/rishabh16196/hf-space-composer.git repo
cd repo

echo "[3/4] pip install"
pip install -q --upgrade pip
pip install -q -e .
pip install -q 'peft>=0.13.0' 'huggingface_hub>=0.26.0' 'safetensors>=0.4.0' 'transformers>=4.46.0'

ADAPTER_REPO="${ADAPTER_REPO:-rishabh16196/spaces-pipeline-grpo-1.5b-v3-search-aware}"
echo "[4/4] download adapter $ADAPTER_REPO and run eval"
hf download "$ADAPTER_REPO" --local-dir /tmp/adapter
cd local_training
python -u eval_two_tier.py \
  --model Qwen/Qwen2.5-1.5B-Instruct \
  --adapter /tmp/adapter

echo "=== ALL DONE ==="
