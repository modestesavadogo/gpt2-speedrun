# Build Log — GPT-2 Speedrun

Tracking the incremental build: vanilla GPT-2 -> RoPE -> QK-norm -> ReLU^2 -> Muon.
Following the nanoGPT / llm.c / modded-nanogpt lineage, adapted for Kaggle
P100/T4 (no flash attention, no fp8, no bf16 -- Pascal/Turing don't support
them, so this uses fp16 + GradScaler instead).

Goal: reproduce, understand, then modernize one technique at a time, with
a before/after loss comparison for each. Each technique = one commit.

---

## Day 1 — Baseline (done, partial run)
- Architecture: vanilla GPT-2 (learned pos embeddings, LayerNorm, GELU MLP)
- GPU: Kaggle T4 (P100 turned out incompatible with Kaggle's current PyTorch build — sm_60 not supported, sm_70+ required)
- Precision: fp16 + GradScaler
- Steps: 250 / 3000 planned (stopped early — pipeline confirmed working, full run deferred to a later comparison pass)
- Tokens seen: ~32.8M
- Train loss: 5.6495
- Val loss: 5.7123
- Wall time: 42.7 min (~10.2s/step)
- Notes: flash_sdp_enabled() printed True on T4 (my prediction of False was wrong — that flag just reports whether flash is toggled on globally, not hardware support; ignore my earlier claim about it)

## Day 2 — RoPE (done)
- Change: removed learned positional embeddings (wpe), added rotary embeddings applied to q/k inside every attention layer
- GPU: Kaggle T4, same config as Day 1 (configs/kaggle_p100.py)
- Steps: 250 (same checkpoint as Day 1, for direct comparison)
- Val loss: 5.2801 (vs. 5.7123 baseline — 0.43 lower)
- Train loss: 5.2064 (vs. 5.6495 baseline)
- Wall time: ~11.1s/step (vs. ~10.2s/step baseline — ~9% slower, expected: rotation adds compute per attention call)
- Param count: 123.65M vs 124.44M baseline (exactly matches the removed wpe table: 1024 x 768 = 786,432 params)
- Caveat: 250 steps is a smoke test, not a scaling claim — direction is consistent with published RoPE results, but this alone doesn't prove it generalizes

## Day 3 — QK-norm + ReLU^2 (done)
- Changes: RMSNorm applied to q/k right before the attention dot product (after RoPE rotation); MLP activation swapped from GELU to squared ReLU
- GPU: Kaggle T4, same config, same 250-step checkpoint for comparison
- Val loss: 5.2474 (vs. 5.2801 RoPE-only, vs. 5.7123 baseline)
- Train loss: 5.1919
- Wall time: ~12.5s/step (vs. ~11.1s/step RoPE-only — ~13% slower, expected: extra norm ops + squaring)
- Caveat: smaller effect than RoPE, and at only 250 steps this could partly be run-to-run noise rather than a robust signal — direction is consistent with Primer's and the QK-norm papers' findings, but this single short run doesn't establish that on its own

## Day 4 — Muon optimizer (done)
- Change: 2D weight matrices in attention/MLP (48 tensors, 84.9M params) moved from AdamW to Muon (Newton-Schulz orthogonalized updates); tied embedding (38.6M params) and all norms/biases (121K params) stay on AdamW
- Val loss: 4.8146 (vs. 5.2474 prior day — 0.43 lower, biggest single-day drop since RoPE)
- Train loss: 4.7338
- Wall time: ~13.6s/step (vs. ~12.5s/step — ~9% slower, extra Newton-Schulz matmuls)
- Sanity check: 84,934,656 + 38,597,376 + 121,344 = 123,653,376 ≈ 123.65M total params — confirms no parameter silently dropped between optimizers
- Simplification: Muon lr fixed at 0.02, no warmup/cosine schedule yet — only AdamW group follows the schedule. Worth revisiting if pushing to a longer real run later.

## Day 5 — Writeup (not started)
Turn this log into a short technical article: one paragraph per technique,
why it works, what it measurably bought. Draft for Medium.
