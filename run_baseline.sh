#!/bin/bash
# Run GPT-2 Small baseline on FineWeb (~100M tokens, ~1 hour on RTX 4060)
# Set DATA_DIR to where your FineWeb .bin files are, or use TinyShakespeare by default

# Update this to point at your FineWeb data
DATA_DIR="${DATA_DIR:-../modded-nanogpt/data/fineweb10B}"

if ls "$DATA_DIR"/fineweb_train_*.bin 1> /dev/null 2>&1; then
    echo "Using FineWeb data from $DATA_DIR"
    python train_gpt2.py \
      --input_bin "$DATA_DIR/fineweb_train_*.bin" \
      --input_val_bin "$DATA_DIR/fineweb_val_*.bin" \
      --batch_size 4 \
      --sequence_length 1024 \
      --total_batch_size 131072 \
      --num_iterations 800 \
      --warmup_iters 50 \
      --learning_rate 3e-4 \
      --weight_decay 0.1 \
      --grad_clip 1.0 \
      --val_loss_every 50 \
      --val_max_steps 10 \
      --tensorcores 1 \
      --dtype bfloat16 \
      --compile 1 \
      --model d12 \
      --overfit_single_batch 0
else
    echo "No FineWeb data found at $DATA_DIR. Running on TinyShakespeare for quick test."
    python train_gpt2.py \
      --sequence_length 256 \
      --num_iterations 50 \
      --warmup_iters 10 \
      --tensorcores 1 \
      --dtype bfloat16 \
      --compile 1 \
      --model d12 \
      --overfit_single_batch 0
fi
