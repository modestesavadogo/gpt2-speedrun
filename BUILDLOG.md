# Build Log — GPT-2 Speedrun

Tracking the incremental build: vanilla GPT-2 -> RoPE -> QK-norm -> ReLU^2 -> Muon.
Following the nanoGPT / llm.c / modded-nanogpt lineage, adapted for Kaggle
P100/T4 (no flash attention, no fp8, no bf16 -- Pascal/Turing don't support
them, so this uses fp16 + GradScaler instead).

Goal: reproduce, understand, then modernize one technique at a time, with
a before/after loss comparison for each. Each technique = one commit.

---

## Day 1 — Baseline (pending)
- Architecture: vanilla GPT-2 (learned pos embeddings, LayerNorm, GELU MLP)
- GPU: Kaggle P100
- Precision: fp16 + GradScaler
- Data: FineWeb-Edu sample, tokenized with tiktoken (gpt2 encoding)
- Status: pipeline sanity check, not the real baseline run yet
- Val loss: TBD
- Notes: TBD (fill in after first run — OOM issues, throughput, anything
  that didn't match expectations)

## Day 2 — RoPE (not started)
Replace learned positional embeddings (wpe) with rotary embeddings,
implemented by hand. Expect: comparable or better loss, better length
generalization (not directly testable here at fixed block_size, but worth
noting in the writeup).

## Day 3 — QK-norm + ReLU^2 (not started)
Normalize queries/keys before the attention dot product; swap GELU MLP
activation for ReLU^2. Both are cheap changes with documented convergence
benefits in the modded-nanogpt ablations.

## Day 4 — Muon optimizer (not started)
Replace AdamW (for matrix parameters only) with Muon — Newton-Schulz
orthogonalization of the gradient. Implement the iteration by hand rather
than importing it. This is the one I expect to actually take real study
time to understand, not just wire up.

## Day 5 — Writeup (not started)
Turn this log into a short technical article: one paragraph per technique,
why it works, what it measurably bought. Draft for Medium.
