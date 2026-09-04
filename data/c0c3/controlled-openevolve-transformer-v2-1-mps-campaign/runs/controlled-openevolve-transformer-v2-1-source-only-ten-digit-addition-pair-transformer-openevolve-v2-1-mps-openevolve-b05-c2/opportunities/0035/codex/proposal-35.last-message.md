MECHANISM: Incremental LayerNorm-input gauge fixing for one QKV row

HYPOTHESIS: Applying one mean-zero QKV-row gauge to the qualified 1539-parameter four-anchor design will produce a 1538-parameter model with at least 99% accuracy, because it removes one LayerNorm-null direction while avoiding the optimization disruption caused by constraining all 24 rows.

INTENDED_EDIT: Extend `ln1` from two to four fixed scales and represent only the final QKV weight row in the seven-dimensional mean-zero basis, preserving full-sized initialization draws.

EVIDENCE: Four anchored scales achieved 99.95% at 1539 parameters, whereas reducing all 24 QKV rows collapsed to 46.48%; this motivates testing the same gauge one row at a time.

<<<<<<< SEARCH
class MeanZeroInputLinear(nn.Linear):
    """Linear map modulo the common input mode removed by LayerNorm."""
    def __init__(self, in_features: int, out_features: int):
        super().__init__(in_features, out_features, bias=True)
        basis = _mean_zero_basis(in_features)
        self.weight = nn.Parameter(self.weight.detach() @ basis)
        self.register_buffer("basis", basis, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = self.weight @ self.basis.transpose(0, 1)
        return F.linear(x, weight, self.bias)


class MeanZeroOutputLinear(nn.Linear):
=======
class MeanZeroInputLinear(nn.Linear):
    """Linear map modulo the common input mode removed by LayerNorm."""
    def __init__(self, in_features: int, out_features: int):
        super().__init__(in_features, out_features, bias=True)
        basis = _mean_zero_basis(in_features)
        self.weight = nn.Parameter(self.weight.detach() @ basis)
        self.register_buffer("basis", basis, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = self.weight @ self.basis.transpose(0, 1)
        return F.linear(x, weight, self.bias)


class SingleRowMeanZeroInputLinear(nn.Module):
    """Bias-free linear map with one LayerNorm-null row direction removed."""
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

        self.weight_rows = nn.Parameter(full_weight[:-1])
        self.last_weight = nn.Parameter(full_weight[-1] @ basis)

    def reset_from_full(self, full_weight: torch.Tensor) -> None:
        with torch.no_grad():
            self.weight_rows.copy_(full_weight[:-1])
            self.last_weight.copy_(full_weight[-1] @ self.basis)

    @property
    def weight(self) -> torch.Tensor:
        last_row = self.basis @ self.last_weight
        return torch.cat((self.weight_rows, last_row.unsqueeze(0)), dim=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight)


class MeanZeroOutputLinear(nn.Linear):
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.qkv.bias = None
=======
        self.qkv = SingleRowMeanZeroInputLinear(d_model, 3 * d_model)
>>>>>>> REPLACE

<<<<<<< SEARCH
class TwiceAnchoredScaleLayerNorm(nn.Module):
    """LayerNorm with two scales fixed through the following linear-map gauge."""
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim - 2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = F.pad(self.weight, (0, 2), value=1.0)
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
        self.ln1 = TwiceAnchoredScaleLayerNorm(cfg.d_model)
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
        elif isinstance(module, SingleRowMeanZeroInputLinear):
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