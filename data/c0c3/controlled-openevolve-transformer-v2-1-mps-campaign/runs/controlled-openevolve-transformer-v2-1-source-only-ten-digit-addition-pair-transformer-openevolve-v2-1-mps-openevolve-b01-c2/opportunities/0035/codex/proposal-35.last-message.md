MECHANISM: Single-coordinate attention LayerNorm scale ablation

HYPOTHESIS: Starting from the verified 1,591-parameter design, fixing one additional `ln1` scale coordinate will produce a 1,590-parameter model with at least 99% accuracy.

INTENDED_EDIT: Make `ln2` affine-free, gauge-fix every `fc1` row, and retain six learned `ln1` scales while fixing its final two scale coordinates to one.

EVIDENCE: The affine-free `ln2` with all-row `fc1` gauge reached 99.96% at 1,591 parameters, whereas removing all seven remaining `ln1` scales fell to 83.94%; removing one scale is the smallest incremental probe of that optimization-sensitive redundancy.

<<<<<<< SEARCH
        self.gauged_rows = (0, 1, 2, 4, 5, 6)
=======
        self.gauged_rows = tuple(range(out_features))
>>>>>>> REPLACE

<<<<<<< SEARCH
class AnchoredScaleLayerNorm(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.normalized_shape = (d_model,)
        self.weight = nn.Parameter(torch.ones(d_model - 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = torch.cat((self.weight, self.weight.new_ones(1)))
        return F.layer_norm(x, self.normalized_shape, weight, None, 1e-5)
=======
class AnchoredScaleLayerNorm(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.normalized_shape = (d_model,)
        self.weight = nn.Parameter(torch.ones(d_model - 2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = torch.cat((self.weight, self.weight.new_ones(2)))
        return F.layer_norm(x, self.normalized_shape, weight, None, 1e-5)
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.ln2 = AnchoredScaleLayerNorm(cfg.d_model)
        self.mlp = MLP(cfg.d_model, cfg.d_ff, cfg.dropout)
=======
        # fc1 absorbs feature scaling; affine-free normalization also makes
        # every fc1 common-row coefficient an exact null direction.
        self.ln2 = nn.LayerNorm(cfg.d_model, elementwise_affine=False)
        self.mlp = MLP(cfg.d_model, cfg.d_ff, cfg.dropout)
>>>>>>> REPLACE