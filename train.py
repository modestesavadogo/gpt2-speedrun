"""
Training loop for the GPT-2 speedrun project.

Key choice for Kaggle P100/T4: fp16 autocast + GradScaler, NOT bf16.
Pascal (P100) has no bf16 tensor core support at all; fp16 does, but fp16's
narrower exponent range means gradients can underflow to zero during
backward, hence GradScaler (it scales the loss up before backward, then
unscales gradients before the optimizer step, so small values don't get
truncated to 0 in fp16). bf16 doesn't need this trick because it has the
same exponent range as fp32 -- just less precision. If I ever move this to
an A100 (Kaggle doesn't offer one, but Colab Pro does), switch to bf16 and
drop the scaler entirely, it's strictly simpler there.

Resumable by design -- Kaggle sessions get cut off, so every checkpoint has
enough state (model, optimizer, step, best val loss) to continue exactly.
"""

import os
import time
import math
import argparse
import importlib

import numpy as np
import torch

from model import GPT, GPTConfig


def get_batch(split, data_dir, block_size, batch_size, device):
    path = os.path.join(data_dir, f"{split}.bin")
    data = np.memmap(path, dtype=np.uint16, mode="r")
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([torch.from_numpy((data[i:i + block_size]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i + 1:i + 1 + block_size]).astype(np.int64)) for i in ix])
    if device == "cuda":
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)
    return x, y


def get_lr(step, warmup_steps, lr_decay_steps, min_lr, max_lr):
    # linear warmup, then cosine decay to min_lr -- standard recipe, nothing exotic yet
    if step < warmup_steps:
        return max_lr * (step + 1) / warmup_steps
    if step > lr_decay_steps:
        return min_lr
    decay_ratio = (step - warmup_steps) / (lr_decay_steps - warmup_steps)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (max_lr - min_lr)


@torch.no_grad()
def estimate_loss(model, data_dir, block_size, batch_size, device, eval_iters, ctx):
    out = {}
    model.eval()
    for split in ["train", "val"]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split, data_dir, block_size, batch_size, device)
            with ctx:
                _, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out


def append_buildlog(entry: str, path="BUILDLOG.md"):
    with open(path, "a") as f:
        f.write(entry + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs.kaggle_p100")
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--out_dir", type=str, default="checkpoints")
    parser.add_argument("--resume", action="store_true",
                         help="resume from out_dir/ckpt.pt if it exists")
    args = parser.parse_args()

    cfg = importlib.import_module(args.config)
    os.makedirs(args.out_dir, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    device_type = "cuda" if "cuda" in device else "cpu"
    print(f"device: {device}")
    if device_type == "cuda":
        print(f"gpu: {torch.cuda.get_device_name(0)}")
        # sanity check, matches what we expect on P100/T4: no flash/mem-efficient
        # backend -> SDPA silently falls back to the math kernel
        print(f"flash sdp available: {torch.backends.cuda.flash_sdp_enabled()}")

    torch.manual_seed(1337)

    model_args = dict(
        n_layer=cfg.n_layer, n_head=cfg.n_head, n_embd=cfg.n_embd,
        block_size=cfg.block_size, bias=cfg.bias, vocab_size=cfg.vocab_size,
        dropout=cfg.dropout,
    )

    start_step = 0
    best_val_loss = float("inf")
    ckpt_path = os.path.join(args.out_dir, "ckpt.pt")

    if args.resume and os.path.exists(ckpt_path):
        print(f"resuming from {ckpt_path}")
        checkpoint = torch.load(ckpt_path, map_location=device)
        gptconf = GPTConfig(**checkpoint["model_args"])
        model = GPT(gptconf)
        model.load_state_dict(checkpoint["model"])
        start_step = checkpoint["step"]
        best_val_loss = checkpoint["best_val_loss"]
    else:
        print("initializing new model from scratch")
        gptconf = GPTConfig(**model_args)
        model = GPT(gptconf)

    model.to(device)

    # fp16 for Pascal/Turing (P100/T4). GradScaler is what makes fp16 safe --
    # see module docstring.
    use_amp = device_type == "cuda"
    amp_dtype = torch.float16
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    import contextlib
    ctx = torch.cuda.amp.autocast(dtype=amp_dtype) if use_amp else contextlib.nullcontext()

    adamw_optimizer, muon_optimizer = model.configure_optimizers(
        cfg.weight_decay, cfg.learning_rate, (cfg.beta1, cfg.beta2), device_type
    )
    if args.resume and os.path.exists(ckpt_path):
        adamw_optimizer.load_state_dict(checkpoint["adamw_optimizer"])
        muon_optimizer.load_state_dict(checkpoint["muon_optimizer"])
    t0 = time.time()
    running_mfu = -1.0

    for step in range(start_step, cfg.max_steps):
        lr = get_lr(step, cfg.warmup_steps, cfg.lr_decay_steps, cfg.min_lr, cfg.learning_rate)
        for param_group in adamw_optimizer.param_groups:
            param_group["lr"] = lr

        # gradient accumulation to hit the target effective batch size on
        # limited VRAM -- P100 has 16GB, can't fit the "real" batch size in
        # one shot at block_size=1024
        adamw_optimizer.zero_grad(set_to_none=True)
        muon_optimizer.zero_grad(set_to_none=True)
        accum_loss = 0.0
        for micro_step in range(cfg.grad_accum_steps):
            X, Y = get_batch("train", args.data_dir, cfg.block_size, cfg.batch_size, device)
            with ctx:
                logits, loss = model(X, Y)
                loss = loss / cfg.grad_accum_steps
            scaler.scale(loss).backward()
            accum_loss += loss.item()

        scaler.unscale_(adamw_optimizer)
        scaler.unscale_(muon_optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        scaler.step(adamw_optimizer)
        scaler.step(muon_optimizer)
        scaler.update()

        if step % cfg.log_interval == 0:
            dt = time.time() - t0
            print(f"step {step}: loss {accum_loss:.4f}, lr {lr:.2e}, {dt:.1f}s elapsed")

        if step > 0 and step % cfg.eval_interval == 0:
            losses = estimate_loss(
                model, args.data_dir, cfg.block_size, cfg.batch_size,
                device, cfg.eval_iters, ctx,
            )
            print(f"step {step}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

            if losses["val"] < best_val_loss:
                best_val_loss = losses["val"]

            checkpoint = {
                "model": model.state_dict(),
                "adamw_optimizer": adamw_optimizer.state_dict(),
                "muon_optimizer": muon_optimizer.state_dict(),
                "model_args": model_args,
                "step": step,
                "best_val_loss": best_val_loss,
                "config": vars(cfg) if not isinstance(cfg, dict) else cfg,
            }
            torch.save(checkpoint, ckpt_path)
            print(f"checkpoint saved to {ckpt_path}")

    # final eval + buildlog entry
    losses = estimate_loss(
        model, args.data_dir, cfg.block_size, cfg.batch_size, device, cfg.eval_iters, ctx
    )
    total_time = time.time() - t0
    tokens_seen = cfg.max_steps * cfg.grad_accum_steps * cfg.batch_size * cfg.block_size

    entry = (
        f"\n## Run @ {time.strftime('%Y-%m-%d %H:%M')}\n"
        f"- GPU: {torch.cuda.get_device_name(0) if device_type == 'cuda' else 'cpu'}\n"
        f"- Precision: fp16 (GradScaler)\n"
        f"- Steps: {cfg.max_steps}, tokens seen: ~{tokens_seen/1e6:.1f}M\n"
        f"- Final train loss: {losses['train']:.4f}\n"
        f"- Final val loss: {losses['val']:.4f}\n"
        f"- Wall time: {total_time/60:.1f} min\n"
    )
    append_buildlog(entry)
    print(entry)
    print("done. see BUILDLOG.md for the logged entry.")


if __name__ == "__main__":
    main()
