#!/usr/bin/env bash
set -euo pipefail

echo "=== HF Job: SFT (3 epochs) + GRPO (100 steps) on search-aware data ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
python -c "import torch; print('pre-upgrade torch:', torch.__version__)"
echo

echo "[1/6] apt deps"
apt-get update -qq
apt-get install -y -qq git curl build-essential

echo "[2/6] git clone repo @ HEAD (must contain search-aware heuristic + hardened rubric)"
cd /tmp
git clone --depth=1 https://github.com/rishabh16196/hf-space-composer.git repo
cd repo
git log --oneline -1

echo "[3/6] install torch 2.5.1 + Unsloth (OpenEnv recipe)"
pip install -q --upgrade pip uv
uv pip install --system -q \
    "torch>=2.8.0" "torchvision>=0.25.0" "triton>=3.4.0" bitsandbytes \
    "transformers==4.56.2" \
    "unsloth_zoo[base] @ git+https://github.com/unslothai/unsloth-zoo" \
    "unsloth[base] @ git+https://github.com/unslothai/unsloth"
uv pip install --system --upgrade --no-deps -q \
    "transformers==4.56.2" tokenizers "trl==0.22.2" unsloth unsloth_zoo
pip install -q -e .
pip install -q 'peft>=0.13.0' 'datasets>=3.0.0' 'accelerate>=0.34.0' 'huggingface_hub>=0.26.0' 'safetensors>=0.4.0'
python -c "import torch; print('final torch:', torch.__version__, '| cuda:', torch.cuda.is_available())"
python -c "from unsloth import FastLanguageModel; print('unsloth OK')"

echo "[4/6] verify hardened rubric + search-aware data"
python -c "
from server.rubrics import compute_format_score
h = {k: 'synthetic_value' for k in 'abcdefghijkl'}
assert compute_format_score(h, {k: 'str' for k in h}) < 0.2, 'Fix A not active'
print('hardened rubric: OK')

import json
pairs = [json.loads(l) for l in open('fixtures/sft_pairs.jsonl')]
n_search = sum(1 for p in pairs if 'search_spaces' in p['completion'])
print(f'SFT pairs: {len(pairs)}  (search actions: {n_search}, {100*n_search/len(pairs):.1f}%)')
assert n_search > 100, 'search-aware data not loaded'
print('search-aware data: OK')
"

echo "[5/6] run SFT (3 epochs on search-aware data)"
cd local_training
python -u sft_local.py \
  --model Qwen/Qwen2.5-1.5B-Instruct \
  --output-dir outputs/sft_search_aware \
  --epochs 3 \
  --batch-size 2 \
  --grad-accum 4 \
  --lr 2e-4

echo "[6/6] run GRPO (100 steps, hardened rubric, search-aware warmstart)"
python -u grpo_unsloth.py \
  --model Qwen/Qwen2.5-1.5B-Instruct \
  --adapter outputs/sft_search_aware \
  --output-dir /tmp/grpo_out \
  --max-steps 100 \
  --num-gens 8 \
  --batch-size 4 \
  --update-micro-batch 16 \
  --save-every 25 \
  --lr 3e-6 \
  --beta 0.05

echo "=== uploading both adapters ==="
hf upload rishabh16196/spaces-pipeline-sft-1.5b-search-aware outputs/sft_search_aware \
  --commit-message "SFT 1.5B search-aware, 3 epochs on 651 search-including pairs"
hf upload rishabh16196/spaces-pipeline-grpo-1.5b-v3-search-aware /tmp/grpo_out \
  --commit-message "GRPO v3 search-aware: warmstart from SFT search-aware + hardened rubric"

echo "=== ALL DONE ==="
