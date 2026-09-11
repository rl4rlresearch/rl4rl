MECHANISM: Fixed three-microbatch optimizer amortization

HYPOTHESIS: Restoring ReLU-squared MLPs and increasing the fixed batch to 786,432 tokens will amortize optimizer overhead enough to process more than 497M tokens while retaining sufficient updates, lowering val_bpb below 0.995558.

INTENDED_EDIT: Restore the proven 4× ReLU-squared MLP and accumulate three 128-sequence microbatches per optimizer step.

EVIDENCE: The fixed two-microbatch design achieved 0.995558 on 497.0M tokens, outperforming the fixed one-microbatch design’s 0.996687 on 408.9M tokens; this motivates testing one additional fixed microbatch without the irregular schedule that fell to 393.2M tokens.

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
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=False)
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=False)

    def forward(self, x):
        x = self.c_fc(x)
        x = F.relu(x).square()
        x = self.c_proj(x)
        return x
>>>>>>> REPLACE

<<<<<<< SEARCH
TOTAL_BATCH_SIZE = 2**19 # ~524K tokens per optimizer step
=======
TOTAL_BATCH_SIZE = 3 * 2**18 # ~786K tokens; three 128-sequence microbatches per step
>>>>>>> REPLACE