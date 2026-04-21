# Local Training (Apple Silicon / MPS)

Separate package for **local prototyping on macOS**. Does NOT modify the hackathon `scripts/` folder — those target Unsloth + CUDA at the venue.

## Why separate?

- Unsloth doesn't support Mac/MPS
- bitsandbytes 4-bit doesn't support Mac/MPS
- The hackathon `scripts/sft_warmstart.py` assumes Unsloth-first → it works here as a fallback but isn't optimized for MPS quirks

This folder uses plain `transformers` + `peft` + `trl`, tuned for MPS memory constraints.

## Setup (one-time)

```bash
cd local_training
python3.12 -m uv venv .venv --python python3.12
python3.12 -m uv pip install --python .venv/bin/python \
    'torch>=2.4.0' 'transformers>=4.46.0' 'peft>=0.13.0' \
    'trl>=0.13.0' 'datasets>=3.0.0' 'accelerate>=0.34.0'
```

## 1-epoch smoke test (~15 min expected on M-series)

```bash
.venv/bin/python sft_local.py --dry-run                      # validate setup
.venv/bin/python sft_local.py --subset 100                   # mini run on 100 pairs
.venv/bin/python sft_local.py                                # full run, 1 epoch, 771 pairs
```

## Model choices

| Model | Gated? | Params | Approx time (1 epoch, 771 pairs) |
|---|---|---|---|
| `Qwen/Qwen2.5-0.5B-Instruct` (default) | No | 0.5B | ~20-30 min |
| `Qwen/Qwen2.5-1.5B-Instruct` | No | 1.5B | ~60-90 min |
| `meta-llama/Llama-3.2-1B-Instruct` | **Yes** (accept at https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct) | 1B | ~40-60 min |

Swap with `--model <hf_model_id>`.

## Output

LoRA adapter saved to `outputs/sft_local/` — tiny (~30MB), attaches to base model for inference.

## Limitations of local MPS

- No 4-bit quant → higher memory usage
- Generation during GRPO rollouts is slow → GRPO not recommended here
- No `flash-attn` on MPS → slower attention
- `bf16` works but slower than CUDA bf16

For real GRPO training, use `../scripts/train_grpo.py` on a Linux+CUDA box at the venue.
