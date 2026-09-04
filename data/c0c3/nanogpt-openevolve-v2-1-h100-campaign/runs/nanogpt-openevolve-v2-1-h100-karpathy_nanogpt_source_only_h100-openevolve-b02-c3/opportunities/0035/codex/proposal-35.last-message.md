MECHANISM: Recursive per-layer lower-half feed-forward taper

HYPOTHESIS: A 2×/2.5×/2.5×/3× lower-layer MLP ramp with uniform 5.5× upper MLPs will process at least 500M tokens and reduce val_bpb below 0.991682.

INTENDED_EDIT: Preserve the best design’s lower-quarter averages and total MLP capacity while redistributing capacity within each lower-layer pair toward greater depth.

EVIDENCE: The 2.25×/2.75× lower-half taper achieved the best observed val_bpb of 0.991682, outperforming both uniform 2.5× and stronger pairwise tapers; recursively applying the same ±0.25× shift within those pairs tests smoother depth localization without changing aggregate parameters or MLP compute.

<<<<<<< SEARCH
class MLP(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        if layer_idx < config.n_layer // 4:
            mlp_mult_halves = 3   # 1.5x
        elif layer_idx < config.n_layer // 2:
            mlp_mult_halves = 7   # 3.5x
        else:
            mlp_mult_halves = 11  # 5.5x
        mlp_dim = mlp_mult_halves * config.n_embd // 2
        self.c_fc = nn.Linear(config.n_embd, mlp_dim, bias=False)
        self.c_proj = nn.Linear(mlp_dim, config.n_embd, bias=False)
=======
class MLP(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        if layer_idx == 0:
            mlp_mult_halves = 4   # 2x
        elif layer_idx < config.n_layer // 2 - 1:
            mlp_mult_halves = 5   # 2.5x
        elif layer_idx < config.n_layer // 2:
            mlp_mult_halves = 6   # 3x
        else:
            mlp_mult_halves = 11  # 5.5x
        mlp_dim = mlp_mult_halves * config.n_embd // 2
        self.c_fc = nn.Linear(config.n_embd, mlp_dim, bias=False)
        self.c_proj = nn.Linear(mlp_dim, config.n_embd, bias=False)
>>>>>>> REPLACE