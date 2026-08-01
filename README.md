# gpt2-speedrun

Reproducing GPT-2 (124M) from the nanoGPT / llm.c / modded-nanogpt lineage,
on Kaggle P100/2xT4 instead of 8xH100 — so no flash attention, no fp8, no
bf16 (Pascal/Turing don't support them). Building each modernization
(RoPE, QK-norm, ReLU², Muon optimizer) by hand, one at a time, instead of
importing the final speedrun script, so each one is a measured before/after
rather than a black box.

Progress tracked in [`BUILDLOG.md`](./BUILDLOG.md) — one entry per
technique, with the loss delta it bought.

This is a reproduction/engineering project, not original research: the
goal is to actually understand each piece of the modern GPT training
recipe by implementing and measuring it, not to set a new record.

## Setup (Kaggle)

1. New notebook -> Settings -> Accelerator: GPU P100 (or 2x T4). Internet: On.
2. Add a GitHub PAT as a Kaggle Secret if you want the notebook to push
   commits/checkpoints directly.
3. First cell:

```bash
!git clone https://github.com/<you>/gpt2-speedrun.git
%cd gpt2-speedrun
!pip install -r requirements.txt --quiet
```

## Usage

```bash
# 1. tokenize a data slice (defaults to 50M tokens, FineWeb-Edu sample)
python prepare_data.py --num_tokens 50_000_000 --out_dir data

# 2. train (fp16 on P100/T4, checkpoints to checkpoints/ckpt.pt)
python train.py --config configs.kaggle_p100 --data_dir data --out_dir checkpoints

# 3. if a Kaggle session gets cut off mid-run, resume:
python train.py --config configs.kaggle_p100 --data_dir data --out_dir checkpoints --resume
```

Each run appends a summary block to `BUILDLOG.md` automatically.

## Why P100/T4 changes the plan

`modded-nanogpt` assumes an 8xH100 node and leans on FlexAttention + FP8,
neither of which exist on Pascal (P100) or Turing (T4) — no sm80+ tensor
core support, no bf16 on P100 either. Everything here is plain PyTorch
(`F.scaled_dot_product_attention` falls back to the `math` kernel on this
hardware, confirmed via `torch.backends.cuda.flash_sdp_enabled()` at
startup) so it runs correctly, just slower than the H100 numbers in the
original repos. The point isn't to match their wall-clock records — it's
to implement the same techniques and see the effect for myself.

## Repo layout

```
model.py            GPT-2 architecture (nanoGPT-style)
prepare_data.py      tokenizes FineWeb-Edu -> train.bin / val.bin
train.py             training loop, fp16/GradScaler, resumable checkpoints
configs/
  kaggle_p100.py     hyperparams sized for 16GB VRAM
BUILDLOG.md           trace of advancement, one entry per technique
```

## References

- [nanoGPT](https://github.com/karpathy/nanoGPT) — Karpathy
- [llm.c](https://github.com/karpathy/llm.c) — Karpathy
- [modded-nanogpt](https://github.com/KellerJordan/modded-nanogpt) — Keller Jordan et al., the speedrun this follows
