# NanoGPT Baseline (from karpathy/llm.c)

**Single-file GPT-2 training.** Targets the same 3.28 val loss on FineWeb as modded-nanogpt, but standard PyTorch — works on any GPU (RTX 4060 included). No flashy tricks, just clean GPT-2.

## What to Modify

The forward pass lives in two places:

- **`Block.forward()`** (line 112-115) — one layer's computation. Currently a standard pre-LN transformer block:
  ```python
  x = x + self.attn(self.ln_1(x))
  x = x + self.mlp(self.ln_2(x))
  ```
  This is where hyperconnections would replace the residual stream.

- **`GPT.forward()`** (line 162-190) — the full model forward pass. The loop over blocks is at line 173-174:
  ```python
  for block in self.transformer.h:
      x = block(x)
  ```

## Running on RTX 4060

**Step 1: Get the data** (from modded-nanogpt or directly):
```bash
# If you have modded-nanogpt checked out alongside this repo:
cd ../modded-nanogpt
pip install -r data/requirements.txt
python data/cached_fineweb10B.py 2  # ~200M tokens, ~2GB

# Or download directly from HuggingFace:
# huggingface-cli download kjj0/fineweb10B-gpt2 --local-dir ./data/fineweb10B
```

**Step 2: Run the baseline** (GPT-2 Small, ~100M tokens, ~1 hour on a 4060):
```bash
cd /root/nanogpt-baseline
pip install tiktoken
bash run_baseline.sh
```

**Step 3: Run with hyperconnections** — same command, just modify `train_gpt2.py` and watch if val loss drops faster.

## Comparison Method

1. Run the baseline → note val loss at each checkpoint (logged to console)
2. Modify `Block.forward()` with hyperconnections
3. Run again with same args → compare val loss at same step

Lower val loss at same step count = improvement. The FineWeb val set is deterministic, so differences of ~0.01 are real.

## If You Want Faster Iteration

For quick tests, use TinyShakespeare (built-in, ~1M tokens, 10 seconds per run):
```bash
python train_gpt2.py --overfit_single_batch 0 --num_iterations 100 --sequence_length 256 --tensorcores 1 --compile 1 --dtype bfloat16
```
But TinyShakespeare doesn't predict FineWeb val loss, so only use this for debugging your implementation.

## Skills

### /study
**Description**: Hands-off learning mode. Execute requests literally without proactive guidance.
**Instructions**:
- When asked to run something that breaks: just show the raw log/error output, don't explain the cause or suggest fixes.
- When asked about code that is wrong or broken: just answer the specific question, don't point out the issue or warn about it.
- Never proactively suggest improvements, optimizations, or alternative approaches unless explicitly asked.
- Never warn about potential pitfalls, edge cases, or downstream effects of a change.
- The user's job is to discover problems themselves — your job is to be a tool, not a tutor.

## Key Architecture Details

- GPT-2 Small: 12 layers, 12 heads, 768 hidden dim, 50257 vocab
- Pre-layer-norm transformer (LayerNorm before each sublayer)
- GELU activation
- Weight tying (embed ↔ lm_head)
- Standard learned positional embeddings
- AdamW optimizer with cosine LR decay
