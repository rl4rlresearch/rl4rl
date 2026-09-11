MECHANISM: Top-quarter-concentrated ReLU² feed-forward capacity

HYPOTHESIS: Preserving the best 2.5× lower-half allocation while shifting upper-half capacity from 5.5× uniformly to 5×/6× by depth will retain at least 495M tokens and reduce val_bpb below 0.991816.

INTENDED_EDIT: Use 2.5× MLPs in layers 0–3, 5× in layers 4–5, and 6× in layers 6–7, preserving the proven 4× average width, optimizer, attention geometry, and schedule.

EVIDENCE: The 2.5×/5.5× taper achieved the best observed 0.991816 val_bpb at 509.1M tokens, after progressively shifting capacity upward improved every tested taper through that point; redistributing the same upper-half capacity toward the final layers tests whether that depth bias continues without increasing aggregate parameters or MLP compute.

<<<<<<< SEARCH
class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        mlp_dim = 9 * config.n_embd // 2
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
        self.mlp = MLP(config)
=======
class MLP(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        if layer_idx < config.n_layer // 2:
            mlp_mult_halves = 5   # 2.5x
        elif layer_idx < 3 * config.n_layer // 4:
            mlp_mult_halves = 10  # 5x
        else:
            mlp_mult_halves = 12  # 6x
        mlp_dim = mlp_mult_halves * config.n_embd // 2
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