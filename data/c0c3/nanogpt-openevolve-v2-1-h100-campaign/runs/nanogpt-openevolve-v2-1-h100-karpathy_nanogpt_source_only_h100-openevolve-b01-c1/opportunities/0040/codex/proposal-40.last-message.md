MECHANISM: Parameter-matched SwiGLU feature routing

HYPOTHESIS: Replacing the additive ReLU² MLP with a parameter-matched multiplicative SwiGLU will lower val_bpb below 0.983689 by allowing each token to conditionally gate predictive features while preserving nearly the same parameter and compute budget.

INTENDED_EDIT: Factor each MLP expansion into learned gate and value branches, with a 64-aligned hidden width chosen to approximately match the existing MLP parameter count.

EVIDENCE: Further static context widening plateaued or regressed—MMMLSSSL reached 0.984942 and concentrating 768 tokens at layer 3 reached 0.984337—suggesting the load-bearing assumption that attention span is the remaining bottleneck should be challenged in favor of more expressive tokenwise feature computation.

<<<<<<< SEARCH
class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=False)
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=False)

    def forward(self, x):
        x = self.c_fc(x)
        x = F.relu(x).square()
        x = self.c_proj(x)
        return x
=======
class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        # SwiGLU uses three hidden-width matrices; round to preserve the
        # parameter budget and Tensor Core-friendly dimensions.
        hidden_dim = ((8 * config.n_embd // 3 + 32) // 64) * 64
        self.c_fc = nn.Linear(config.n_embd, 2 * hidden_dim, bias=False)
        self.c_proj = nn.Linear(hidden_dim, config.n_embd, bias=False)

    def forward(self, x):
        gate, value = self.c_fc(x).chunk(2, dim=-1)
        x = F.silu(gate) * value
        x = self.c_proj(x)
        return x
>>>>>>> REPLACE