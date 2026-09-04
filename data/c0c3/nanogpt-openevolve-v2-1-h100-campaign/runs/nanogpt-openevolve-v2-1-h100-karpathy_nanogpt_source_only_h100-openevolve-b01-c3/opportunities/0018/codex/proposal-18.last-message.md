MECHANISM: Depth-tapered MLP capacity allocation

HYPOTHESIS: A top-heavy 4.25×/4.5× MLP taper with the same aggregate width as the best uniform 4.375× design will process at least 465M tokens and reduce val_bpb below 0.994296.

INTENDED_EDIT: Use 2176-channel MLPs in the first four layers and 2304-channel MLPs in the final four layers.

EVIDENCE: Uniform 4.375× achieved 0.994296, outperforming both 4.25× at 0.994753 and 4.5× at 0.994680; holding the proven aggregate capacity fixed while concentrating it in later layers isolates whether depth-wise allocation improves its value.

<<<<<<< SEARCH
class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        hidden_dim = 17 * config.n_embd // 4
        self.c_fc = nn.Linear(config.n_embd, hidden_dim, bias=False)
        self.c_proj = nn.Linear(hidden_dim, config.n_embd, bias=False)
=======
class MLP(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        expansion_quarters = 17 if layer_idx < config.n_layer // 2 else 18
        hidden_dim = expansion_quarters * config.n_embd // 4
        self.c_fc = nn.Linear(config.n_embd, hidden_dim, bias=False)
        self.c_proj = nn.Linear(hidden_dim, config.n_embd, bias=False)
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.attn = CausalSelfAttention(config, layer_idx)
        self.mlp = MLP(config)
=======
        self.attn = CausalSelfAttention(config, layer_idx)
        self.mlp = MLP(config, layer_idx)
>>>>>>> REPLACE