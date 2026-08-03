"""
GPT-2 architecture, built following the nanoGPT lineage (Karpathy) so I can
compare against the published baseline numbers directly.

Day 1: vanilla GPT-2 (learned pos embeddings, LayerNorm, GELU MLP, standard
multi-head attention). Later days will swap pieces of this out one at a time
(RoPE instead of wpe, RMSNorm instead of LayerNorm, ReLU^2 instead of GELU,
QK-norm) — keeping the baseline clean here so each swap is a single visible
diff in the git log, not a rewrite.
"""

import math
import inspect
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F





def precompute_rope(head_dim, seq_len, base=10000, device=None):
    freqs = 1.0 / (base ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    t = torch.arange(seq_len, device=device).float()
    freqs = torch.outer(t, freqs)          # (seq_len, head_dim/2)
    return torch.cos(freqs), torch.sin(freqs)


def apply_rope(x, cos, sin):
    # x: (B, nh, T, hs) -- rotate pairs of dims by position-dependent angle
    x1, x2 = x[..., ::2], x[..., 1::2]
    cos = cos[None, None, :x.size(2), :]
    sin = sin[None, None, :x.size(2), :]
    rotated = torch.stack([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)
    return rotated.flatten(-2)




@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50257  # GPT-2 BPE vocab (tiktoken "gpt2")
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0  # speedrun convention: no dropout, we're not overfitting
    bias: bool = True  # GPT-2 used bias in Linear/LayerNorm; nanoGPT lets you disable


class LayerNorm(nn.Module):
    """LayerNorm with an optional bias, because PyTorch's built-in doesn't
    support bias=False directly."""

    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, x):
        return F.layer_norm(x, self.weight.shape, self.weight, self.bias, 1e-5)


class CausalSelfAttention(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # combined qkv projection, split after — standard nanoGPT trick,
        # one matmul instead of three
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout

        # flash attention via SDPA — on P100/T4 this silently falls back to
        # the "math" backend since there's no flash/mem-efficient kernel for
        # pre-Ampere. Checked with torch.backends.cuda.flash_sdp_enabled()
        # during the baseline run and it was False, as expected — not a bug.
        self.flash = hasattr(F, "scaled_dot_product_attention")
        if not self.flash:
            print("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")
            self.register_buffer(
                "bias",
                torch.tril(torch.ones(config.block_size, config.block_size)).view(
                    1, 1, config.block_size, config.block_size
                ),
            )

    def forward(self, x):
        B, T, C = x.size()

        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)

        if self.flash:
            y = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=None,
                dropout_p=self.dropout if self.training else 0,
                is_causal=True,
            )
        else:
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float("-inf"))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v

        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_dropout(self.c_proj(y))
        return y


class MLP(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class Block(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, x, cos, sin):
        x = x + self.attn(self.ln_1(x), cos, sin)
        x = x + self.mlp(self.ln_2(x))
        return x


class GPT(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config

        self.transformer = nn.ModuleDict(
            dict(
                wte=nn.Embedding(config.vocab_size, config.n_embd),
                drop=nn.Dropout(config.dropout),
                h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
                ln_f=LayerNorm(config.n_embd, bias=config.bias),
            ),
        )
        head_dim = config.n_embd // config.n_head
        cos, sin = precompute_rope(head_dim, config.block_size)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        # weight tying — wte and lm_head share the same matrix (GPT-2 does this,
        # saves ~40M params at this scale and is a well-known regularizer)
        self.transformer.wte.weight = self.lm_head.weight

        self.apply(self._init_weights)
        # scaled init for residual projections, per GPT-2 paper section 2.3
        for pn, p in self.named_parameters():
            if pn.endswith("c_proj.weight"):
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))

        n_params = sum(p.numel() for p in self.parameters())
        print(f"model initialized: {n_params/1e6:.2f}M parameters")

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        device = idx.device
        b, t = idx.size()
        assert t <= self.config.block_size, (
            f"sequence length {t} exceeds block_size {self.config.block_size}"
        )
        tok_emb = self.transformer.wte(idx)
        x = self.transformer.drop(tok_emb)
        for block in self.transformer.h:
            x = block(x, self.rope_cos, self.rope_sin)
        x = self.transformer.ln_f(x)

        if targets is not None:
            logits = self.lm_head(x)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1
            )
        else:
            # inference-time optimization: only compute logits for last position
            logits = self.lm_head(x[:, [-1], :])
            loss = None

        return logits, loss

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        """Split params into decay/no-decay groups — standard nanoGPT recipe.
        Only 2D+ tensors (matmul weights) get weight decay; LayerNorm/bias/
        embeddings don't."""
        param_dict = {pn: p for pn, p in self.named_parameters() if p.requires_grad}
        decay_params = [p for p in param_dict.values() if p.dim() >= 2]
        nodecay_params = [p for p in param_dict.values() if p.dim() < 2]
        optim_groups = [
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": nodecay_params, "weight_decay": 0.0},
        ]
        num_decay = sum(p.numel() for p in decay_params)
        num_nodecay = sum(p.numel() for p in nodecay_params)
        print(f"decayed params: {len(decay_params)} tensors, {num_decay:,} elements")
        print(f"non-decayed params: {len(nodecay_params)} tensors, {num_nodecay:,} elements")

        # fused AdamW if available and on CUDA — small free speedup
        fused_available = "fused" in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == "cuda"
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
        print(f"using fused AdamW: {use_fused}")
        return optimizer

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        for _ in range(max_new_tokens):
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("Inf")
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx
