MECHANISM: Compute-rebalanced 3× ReLU-squared feed-forward layers

HYPOTHESIS: Reducing the proven ReLU-squared MLP expansion from 4× to 3× will process more than 497M tokens while retaining eight-layer SSSL capacity, lowering val_bpb below 0.995558.

INTENDED_EDIT: Replace the slower parameter-matched SwiGLU with a tensor-core-aligned 3× ReLU-squared MLP; preserve the proven attention, batching, optimizer, and schedule.

EVIDENCE: Reference Design 1 reached 0.995558 on 497.0M tokens, while near-parameter-matched SwiGLU processed only 402.7M tokens and worsened to 1.008305. A narrower ReLU-squared MLP directly targets the demonstrated throughput sensitivity with a modest capacity reduction.

<<<<<<< SEARCH
class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        hidden_dim = ((8 * config.n_embd // 3 + 127) // 128) * 128
        self.c_fc = nn.Linear(config.n_embd, 2 * hidden_dim, bias=False)
        self.c_proj = nn.Linear(hidden_dim, config.n_embd, bias=False)

    def forward(self, x):
        value, gate = self.c_fc(x).chunk(2, dim=-1)
        x = value * F.silu(gate)
        x = self.c_proj(x)
        return x
=======
class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        hidden_dim = 3 * config.n_embd
        self.c_fc = nn.Linear(config.n_embd, hidden_dim, bias=False)
        self.c_proj = nn.Linear(hidden_dim, config.n_embd, bias=False)

    def forward(self, x):
        x = self.c_fc(x)
        x = F.relu(x).square()
        x = self.c_proj(x)
        return x
>>>>>>> REPLACE