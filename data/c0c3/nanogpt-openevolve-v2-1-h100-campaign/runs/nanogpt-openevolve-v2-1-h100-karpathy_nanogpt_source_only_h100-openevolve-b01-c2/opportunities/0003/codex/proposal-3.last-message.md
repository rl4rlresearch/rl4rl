MECHANISM: Parameter-matched SwiGLU feed-forward layers

HYPOTHESIS: Replacing ReLU-squared MLPs with near-parameter-matched SwiGLU MLPs will improve per-token modeling enough to beat 0.995558 val_bpb while retaining the winning SSSL attention layout and similar throughput.

INTENDED_EDIT: Use a tensor-core-aligned SwiGLU hidden width near 8/3 of model dimension, keeping MLP parameter count and compute close to the current 4× ReLU-squared design.

EVIDENCE: The current SSSL design achieved the best result, 0.995558 val_bpb on 497.0M tokens; both attention-window changes reduced throughput and worsened validation, motivating an independent capacity-quality change that leaves attention unchanged.

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
        hidden_dim = ((8 * config.n_embd // 3 + 127) // 128) * 128
        self.c_fc = nn.Linear(config.n_embd, 2 * hidden_dim, bias=False)
        self.c_proj = nn.Linear(hidden_dim, config.n_embd, bias=False)

    def forward(self, x):
        value, gate = self.c_fc(x).chunk(2, dim=-1)
        x = value * F.silu(gate)
        x = self.c_proj(x)
        return x
>>>>>>> REPLACE