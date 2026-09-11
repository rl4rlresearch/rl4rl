MECHANISM: Depth-tapered SwiGLU context feature gating

HYPOTHESIS: Replacing additive ReLU² features with parameter-matched multiplicative SwiGLU features, while retaining the successful upper-layer capacity bias, will process at least 490M tokens and reduce val_bpb below 0.992431.

INTENDED_EDIT: Challenge the assumption that wider ungated ReLU² MLPs are the best context representation: use 2× lower-layer and 3.25× upper-layer SwiGLU widths, approximately matching the proven 3×/5× model’s parameter count and compute.

EVIDENCE: The 3×/5× depth taper achieved the best val_bpb, 0.992431, while further uniform widening regressed; this suggests allocation is useful but ungated width scaling is exhausted. SwiGLU tests learned multiplicative feature selection without the repeated sequence-shift operations that reduced throughput to 390.1M tokens.

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
        # Parameter-match the successful 3x/5x ReLU² taper with a gated MLP:
        # SwiGLU uses three hidden-width matrices instead of two.
        mlp_quarters = 8 if layer_idx < config.n_layer // 2 else 13
        mlp_dim = mlp_quarters * config.n_embd // 4
        self.c_fc = nn.Linear(config.n_embd, 2 * mlp_dim, bias=False)
        self.c_proj = nn.Linear(mlp_dim, config.n_embd, bias=False)

    def forward(self, x):
        gate, value = self.c_fc(x).chunk(2, dim=-1)
        x = F.silu(gate) * value
        x = self.c_proj(x)
        return x


class Block(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.attn = CausalSelfAttention(config, layer_idx)
        self.mlp = MLP(config, layer_idx)
>>>>>>> REPLACE