"""
Muon optimizer (Keller Jordan, simplified) — used in modded-nanogpt for the
2D weight matrices inside transformer blocks (attention/MLP projections).
Embeddings, norms, and biases stay on AdamW (see model.py's split in
configure_optimizers) — orthogonalizing an embedding table doesn't make
sense the way it does for a linear map.

Core mechanism: take the momentum buffer (a matrix), orthogonalize it via
Newton-Schulz iteration before applying it as the update, instead of
AdamW's per-element adaptive scaling. Intuition: every singular direction
of the gradient gets roughly the same step size, rather than a step
proportional to its singular value the way plain SGD/Adam would give it.
"""
import torch


def zeropower_via_newtonschulz5(G, steps=5):
    assert G.ndim == 2
    a, b, c = 3.4445, -4.7750, 2.0315
    X = G.bfloat16()
    X = X / (X.norm() + 1e-7)
    transposed = X.size(0) > X.size(1)
    if transposed:
        X = X.T
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * A @ A
        X = a * X + B @ X
    if transposed:
        X = X.T
    return X.float()


class Muon(torch.optim.Optimizer):
    def __init__(self, params, lr=0.02, momentum=0.95, nesterov=True, ns_steps=5):
        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov, ns_steps=ns_steps)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = closure() if closure is not None else None
        for group in self.param_groups:
            lr, momentum = group["lr"], group["momentum"]
            for p in group["params"]:
                g = p.grad
                if g is None:
                    continue
                state = self.state[p]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(g)
                buf = state["momentum_buffer"]
                buf.mul_(momentum).add_(g)
                g = g.add(buf, alpha=momentum) if group["nesterov"] else buf
                g = zeropower_via_newtonschulz5(g, steps=group["ns_steps"])
                scale = max(1.0, p.size(0) / p.size(1)) ** 0.5
                p.add_(g, alpha=-lr * scale)
        return loss
