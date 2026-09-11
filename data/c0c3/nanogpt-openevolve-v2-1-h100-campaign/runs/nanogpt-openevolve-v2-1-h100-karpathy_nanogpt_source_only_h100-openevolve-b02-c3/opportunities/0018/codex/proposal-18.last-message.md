MECHANISM: Depth-tapered feed-forward capacity allocation

HYPOTHESIS: Shifting MLP capacity from the lower four layers to the upper four while preserving total parameters and compute will retain roughly 513M-token throughput and reduce val_bpb below 0.993287.

INTENDED_EDIT: Use 3.5× MLPs in the lower half and 4.5× MLPs in the upper half, preserving the 4× average width, and restore the best verified 50% linear warmdown.

EVIDENCE: Uniform MLP widths improved monotonically from 3× to 4×, showing that feed-forward capacity matters, while uniform 4.5× lost throughput and regressed; reallocating the same aggregate 4× capacity toward later representations tests a new capacity lever without increasing compute, under the best verified 50% schedule.

<<<<<<< SEARCH
class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=False)
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=False)
=======
class MLP(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        mlp_mult_halves = 7 if layer_idx < config.n_layer // 2 else 9
        mlp_dim = mlp_mult_halves * config.n_embd // 2
        self.c_fc = nn.Linear(config.n_embd, mlp_dim, bias=False)
        self.c_proj = nn.Linear(mlp_dim, config.n_embd, bias=False)
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.attn = CausalSelfAttention(config, layer_idx)
        self.mlp = MLP(config)
=======
        self.attn = CausalSelfAttention(config, layer_idx)
        self.mlp = MLP(config, layer_idx)
>>>>>>> REPLACE

<<<<<<< SEARCH
WARMDOWN_RATIO = 0.6    # fraction of time budget for LR warmdown
=======
WARMDOWN_RATIO = 0.5    # fraction of time budget for LR warmdown
>>>>>>> REPLACE