"""
Config for the Day-1 baseline sanity run on Kaggle P100 (16GB VRAM).

Sizing notes:
- block_size=1024, batch_size=8 fits comfortably on P100 with fp16 for the
  full 124M model. Pushed batch_size up until first OOM, then backed off
  to leave headroom for the eval pass.
- grad_accum_steps=16 -> effective batch = 8 * 16 * 1024 ≈ 131k tokens/step.
  Nowhere near the ~0.5M token/step GPT-2 used, deliberately -- this config
  is for the pipeline sanity check (Day 1), not the real baseline run.
  Bump batch_size/grad_accum once this confirms end-to-end and I move to
  the actual multi-hour run.
- max_steps kept small on purpose (this is "does the loss curve look sane",
  not "match the paper's number" -- that's the next run once this passes).
"""

# model
n_layer = 12
n_head = 12
n_embd = 768
block_size = 1024
vocab_size = 50257
bias = True
dropout = 0.0

# batching
batch_size = 8
grad_accum_steps = 16

# optimizer
learning_rate = 6e-4
min_lr = 6e-5
weight_decay = 0.1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0

# schedule
warmup_steps = 100
lr_decay_steps = 3000
max_steps = 3000

# eval / logging
eval_interval = 250
eval_iters = 20
log_interval = 10
