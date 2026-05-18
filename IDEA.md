# Hyperconnections × NanoGPT

## The Idea

[Hyperconnections](https://arxiv.org/abs/2503.03647) are a drop-in replacement for residual connections in transformers. Instead of the standard `x + sublayer(x)` pattern, hyperconnections use a learnable, structured mixing matrix that couples the forward and backward pass across layers, improving gradient flow and training efficiency.

## The Exercise

This repo is an exercise devised by **Claude Code** to experiment with hyperconnections in a controlled, reproducible setting.

**The task:** Take the standard GPT-2 Small baseline in `train_gpt2.py`, replace the residual connections in `Block.forward()` with hyperconnections, and measure whether training improves.

**The measure of success:** Does the validation loss on FineWeb reach a lower value at the same step count compared to the baseline? The FineWeb validation set is deterministic and the target loss (3.28) is the same one used in the modded-nanogpt speedrun, so differences of ~0.01 are meaningful.

## Motivation

Residual connections were a breakthrough — they made deep networks trainable. But the simple `x + f(x)` pattern may not be optimal. Hyperconnections generalize this by allowing the network to learn how information should flow across layers, both forward (during inference) and backward (during gradient propagation).

Instead of feeding each sublayer's output back into a single residual stream, hyperconnections distribute it across multiple "lanes" with learnable mixing coefficients. This gives the optimizer more degrees of freedom to route gradients around the network during backpropagation.

## The Baseline

- GPT-2 Small (124M parameters): 12 layers, 12 heads, 768 hidden dim
- Standard pre-layer-norm transformer with GELU
- AdamW optimizer, cosine LR schedule
- FineWeb dataset, 3.28 val loss target

One forward pass of one layer:

```python
def forward(self, x):
    x = x + self.attn(self.ln_1(x))
    x = x + self.mlp(self.ln_2(x))
    return x
```

The hyperconnections modification replaces these two residual lines with a learnable mixing scheme.

## Running the Experiment

```bash
# Baseline run
bash run_baseline.sh

# Then modify Block.forward(), run again (same args), compare val loss
```

## References

- Hyperconnections paper: https://arxiv.org/abs/2503.03647
- Original baseline code: https://github.com/karpathy/llm.c (train_gpt2.py)
- Modded-nanogpt speedrun: https://github.com/KellerJordan/modded-nanogpt
