MECHANISM: Incremental second-row LayerNorm-input gauge fixing

HYPOTHESIS: Constraining a second QKV weight row to the mean-zero basis will reduce the qualified 1538-parameter design to 1537 parameters while retaining at least 99% accuracy, because it removes one additional LayerNorm-null direction without the optimization disruption observed when all 24 rows were constrained.

INTENDED_EDIT: Reproduce the qualified four-scale `ln1` anchor and extend the successful single-row mean-zero QKV parameterization to exactly two rows, preserving full-sized fresh initialization draws.

EVIDENCE: The one-row QKV gauge achieved 99.77% accuracy at 1538 parameters, while constraining all 24 rows collapsed to 46.48%; this motivates another one-row incremental reduction.

<<<<<<< SEARCH
class MeanZeroOutputLinear(nn.Linear):
=======
class TwoRowMeanZeroInputLinear(nn.Module):
    """Bias-free linear map with two LayerNorm-null row directions removed."""
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        basis = _mean_zero_basis(in_features)
        self.register_buffer("basis", basis, persistent=False)

        full_weight = torch.empty(out_features, in_features)
        nn.init.kaiming_uniform_(full_weight, a=math.sqrt(5))
        discarded_bias = torch.empty(out_features)
        bound = 1.0 / math.sqrt(in_features)
        nn.init.uniform_(discarded_bias, -bound, bound)

        self.weight_rows = nn.Parameter(full_weight[:-2])
        self.last_weights = nn.Parameter(full_weight[-2:] @ basis)

    def reset_from_full(self, full_weight: torch.Tensor) -> None:
        with torch.no_grad():
            self.weight_rows.copy_(full_weight[:-2])
            self.last_weights.copy_(full_weight[-2:] @ self.basis)

    @property
    def weight(self) -> torch.Tensor:
        last_rows = self.last_weights @ self.basis.transpose(0, 1)
        return torch.cat((self.weight_rows, last_rows), dim=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight)


class MeanZeroOutputLinear(nn.Linear):
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.qkv.bias = None
        self.q_bias = nn.Parameter(torch.zeros(d_model))
=======
        self.qkv = TwoRowMeanZeroInputLinear(d_model, 3 * d_model)
        self.q_bias = nn.Parameter(torch.zeros(d_model))
>>>>>>> REPLACE

<<<<<<< SEARCH
class AnchoredScaleLayerNorm(nn.Module):
    """LayerNorm with one scale fixed through the following linear-map gauge."""
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim - 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = F.pad(self.weight, (0, 1), value=1.0)
        return F.layer_norm(x, (self.dim,), weight, None, self.eps)
=======
class QuadrupleAnchoredScaleLayerNorm(nn.Module):
    """LayerNorm with four scales fixed through the following linear-map gauge."""
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim - 4))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = F.pad(self.weight, (0, 4), value=1.0)
        return F.layer_norm(x, (self.dim,), weight, None, self.eps)
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.ln1 = AnchoredScaleLayerNorm(cfg.d_model)
=======
        self.ln1 = QuadrupleAnchoredScaleLayerNorm(cfg.d_model)
>>>>>>> REPLACE

<<<<<<< SEARCH
        elif isinstance(module, MeanZeroInputLinear):
            full_weight = module.weight.new_empty(
                module.out_features, module.in_features
            )
            nn.init.normal_(full_weight, mean=0.0, std=0.02)
            with torch.no_grad():
                module.weight.copy_(full_weight @ module.basis)
                module.bias.zero_()
=======
        elif isinstance(module, TwoRowMeanZeroInputLinear):
            full_weight = module.weight_rows.new_empty(
                module.out_features, module.in_features
            )
            nn.init.normal_(full_weight, mean=0.0, std=0.02)
            module.reset_from_full(full_weight)
        elif isinstance(module, MeanZeroInputLinear):
            full_weight = module.weight.new_empty(
                module.out_features, module.in_features
            )
            nn.init.normal_(full_weight, mean=0.0, std=0.02)
            with torch.no_grad():
                module.weight.copy_(full_weight @ module.basis)
                module.bias.zero_()
>>>>>>> REPLACE