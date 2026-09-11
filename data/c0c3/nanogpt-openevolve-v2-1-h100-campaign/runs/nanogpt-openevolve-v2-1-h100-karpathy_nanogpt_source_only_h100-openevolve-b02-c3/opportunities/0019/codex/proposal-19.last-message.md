MECHANISM: Stronger depth-tapered feed-forward capacity allocation

HYPOTHESIS: Moving from 3.5×/4.5× to 3×/5× lower/upper MLPs while preserving average width, total parameters, and the best 50% linear warmdown will reduce val_bpb below 0.992809 without materially reducing throughput.

INTENDED_EDIT: Allocate 3× MLP width to the lower four layers and 5× width to the upper four layers, and restore the best verified 50% warmdown.

EVIDENCE: The 3.5×/4.5× depth taper improved val_bpb from the uniform 4× design’s 0.993287 to 0.992809 at equal aggregate MLP capacity, showing that shifting feed-forward capacity upward is beneficial; this patch tests a stronger shift along the same axis.

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


class Block(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.attn = CausalSelfAttention(config, layer_idx)
        self.mlp = MLP(config)
=======
class MLP(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        mlp_mult = 3 if layer_idx < config.n_layer // 2 else 5
        mlp_dim = mlp_mult * config.n_embd
        self.c_fc = nn.Linear(config.n_embd, mlp_dim, bias=False)
        self.c_proj = nn.Linear(mlp_dim, config.n_embd, bias=False)

    def forward(self, x):
        x = self.c_fc(x)
        x = F.relu(x).square()
        x = self.c_proj(x)
        return x


class Block(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.attn = CausalSelfAttention(config, layer_idx)
        self.mlp = MLP(config, layer_idx)
>>>>>>> REPLACE

<<<<<<< SEARCH
WARMDOWN_RATIO = 0.4    # fraction of time budget for LR warmdown
=======
WARMDOWN_RATIO = 0.5    # fraction of time budget for LR warmdown
>>>>>>> REPLACE