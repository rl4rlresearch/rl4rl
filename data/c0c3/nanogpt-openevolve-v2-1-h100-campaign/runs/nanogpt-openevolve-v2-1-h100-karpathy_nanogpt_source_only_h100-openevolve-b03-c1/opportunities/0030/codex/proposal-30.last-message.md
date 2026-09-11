MECHANISM: Variance-matched SwiGLU feature synthesis

HYPOTHESIS: Replacing the homogeneous squared-ReLU feature bank with an approximately parameter-matched multiplicative SwiGLU bank will reduce val_bpb below 0.984227 by learning context-dependent feature interactions without materially reducing token throughput.

INTENDED_EDIT: Challenge the assumption that each MLP should independently activate polynomial features: split its expansion into learned gate and value streams, combine them multiplicatively, and preserve activation scale with a factor of two.

EVIDENCE: The added attention-head gate processed only 480.5M tokens and did not improve quality, while the third global layer improved val_bpb by just 0.000085; this motivates changing the token representation mechanism using a compute-matched replacement rather than adding another attention-side branch.

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
        # 3 * d_model * hidden_dim approximately matches the original 8 * d_model^2.
        # Round to a tensor-core-friendly multiple while retaining nearly identical compute.
        hidden_dim = ((8 * config.n_embd // 3 + 32) // 64) * 64
        self.c_fc = nn.Linear(config.n_embd, 2 * hidden_dim, bias=False)
        self.c_proj = nn.Linear(hidden_dim, config.n_embd, bias=False)

    def forward(self, x):
        gate, value = self.c_fc(x).chunk(2, dim=-1)
        x = 2.0 * F.silu(gate) * value
        x = self.c_proj(x)
        return x
>>>>>>> REPLACE