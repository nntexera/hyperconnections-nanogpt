# NanoGPT Baseline

A clean, single-file GPT-2 training script for benchmarking architecture changes.

This is the reference PyTorch trainer from [karpathy/llm.c](https://github.com/karpathy/llm.c) — standard GPT-2, standard PyTorch, no distributed magic, no FP8, no custom kernels. It targets the same 3.28 validation loss on [FineWeb](https://huggingface.co/datasets/HuggingFaceFW/fineweb) as the modded-nanogpt speedrun, but works on any GPU (RTX 4060 included) out of the box.

## Quick Start

```bash
# 1. Get the data (download 2 shards = 200M tokens)
cd /root/modded-nanogpt
pip install -r data/requirements.txt
python data/cached_fineweb10B.py 2

# 2. Run training baseline (~1 hour on RTX 4060)
cd /root/nanogpt-baseline
pip install tiktoken
bash run_baseline.sh
```

## Modifying the Architecture

The forward pass you want to change is in `Block.forward()` (line 112-115 of `train_gpt2.py`):

```python
def forward(self, x):
    x = x + self.attn(self.ln_1(x))
    x = x + self.mlp(self.ln_2(x))
    return x
```

Replace the residual connections with your own pattern (hyperconnections, etc.) and run again with the same command. Lower validation loss at the same step count = improvement.

## Parameters

```
--model d12            # GPT-2 Small (124M params)
--batch_size 4         # micro-batch size per step
--sequence_length 1024 # context length
--total_batch_size 131072  # effective batch size via grad accumulation
--num_iterations 800   # total training steps
--val_loss_every 50    # checkpoint frequency
--dtype bfloat16       # mixed precision
--compile 1            # torch.compile (~2min warmup, then 2x faster)
```

For faster iteration during development, use TinyShakespeare (built-in, no download needed):
```bash
python train_gpt2.py --overfit_single_batch 0 --num_iterations 100 --sequence_length 256
```

## Architecture

Standard GPT-2 Small:
- 12 layers, 12 heads, 768 hidden dim
- Pre-layer-norm (LayerNorm before each sublayer)
- GELU activation, weight tying, learned positional embeddings
- AdamW optimizer with cosine LR decay
