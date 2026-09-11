MECHANISM: Parameter-neutral SwiGLU feature-selective prediction blocks

HYPOTHESIS: Replacing the single-branch ReLU² MLPs with 64-aligned, nearly parameter-neutral SwiGLU blocks will lower val_bpb below 0.991177 while retaining at least 470M trained tokens.

INTENDED_EDIT: Keep the proven 704/1024/1344 attention backbone and 0.0115 output-gate AdamW rate, but factor every MLP expansion into learned gate and value streams whose multiplicative interaction controls features at each token.

EVIDENCE: Window-allocation and output-gate learning-rate refinements plateaued around the 0.991177 best result, while input-dependent multiplicative output gating was the clearest successful architectural lever. This challenges the shared assumption that an ungated ReLU² feature expansion is sufficient, without increasing the attention budget; the 1344-wide hidden dimension is within 1.6% of iso-parameter SwiGLU sizing and remains tensor-core aligned.

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
        # Three SwiGLU matrices at width 8d/3 match the parameter cost of the
        # two 4d ReLU-squared matrices. Round to 64 for efficient BF16 GEMMs.
        hidden_dim = ((8 * config.n_embd // 3 + 32) // 64) * 64
        self.c_fc = nn.Linear(config.n_embd, 2 * hidden_dim, bias=False)
        self.c_proj = nn.Linear(hidden_dim, config.n_embd, bias=False)

    def forward(self, x):
        gate, value = self.c_fc(x).chunk(2, dim=-1)
        x = F.silu(gate) * value
        x = self.c_proj(x)
        return x
>>>>>>> REPLACE