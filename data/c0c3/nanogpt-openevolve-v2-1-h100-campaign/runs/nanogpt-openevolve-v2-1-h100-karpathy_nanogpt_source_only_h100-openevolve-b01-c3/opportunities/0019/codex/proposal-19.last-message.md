MECHANISM: Bottom-heavy MLP capacity allocation

HYPOTHESIS: Shifting the best 4.375× aggregate MLP capacity toward the first four layers will process at least 465M tokens and reduce val_bpb below 0.994296.

INTENDED_EDIT: Use 2304-channel MLPs in the first four layers and 2176-channel MLPs in the final four layers.

EVIDENCE: The top-heavy allocation worsened val_bpb from 0.994296 to 0.994503 at nearly identical throughput and parameter count, suggesting that removing capacity from early layers was harmful and motivating the inverse allocation.

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
        expansion_quarters = 18 if layer_idx < config.n_layer // 2 else 17
        hidden_dim = expansion_quarters * config.n_embd // 4
        self.c_fc = nn.Linear(config.n_embd, hidden_dim, bias=False)
        self.c_proj = nn.Linear(hidden_dim, config.n_embd, bias=False)
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.mlp = MLP(config)
=======
        self.mlp = MLP(config, layer_idx)
>>>>>>> REPLACE