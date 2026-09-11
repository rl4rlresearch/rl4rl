MECHANISM: LayerNorm-input gauge fixing for QKV weights

HYPOTHESIS: Combining the qualified four-coordinate `ln1` anchor with a mean-zero input-row parameterization for `qkv` will produce a 1515-parameter model with at least 99% accuracy, because each QKV output row retains a complete gauge representative on centered LayerNorm features while the learned scales remain available for optimization.

INTENDED_EDIT: Use the qualified four-scale anchor and remove one redundant input-direction parameter from each of the 24 QKV output rows, preserving full-sized initialization draws and the initialized function.

EVIDENCE: Four anchored `ln1` scales achieved 99.95% at 1539 parameters, while the analogous mean-zero `fc1` input reduction removed 12 LayerNorm-null directions and achieved 99.83%; the fifth scale anchor’s 73.04% failure motivates retaining four learned scales and reducing linear-map gauge directions instead.

<<<<<<< SEARCH
class MeanZeroInputLinear(nn.Linear):
    """Linear map modulo the common input mode removed by LayerNorm."""
    def __init__(self, in_features: int, out_features: int):
        super().__init__(in_features, out_features, bias=True)
        basis = _mean_zero_basis(in_features)
        self.weight = nn.Parameter(self.weight.detach() @ basis)
        self.register_buffer("basis", basis, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight @ self.basis.transpose(0, 1), self.bias)
=======
class MeanZeroInputLinear(nn.Linear):
    """Linear map with one LayerNorm input-direction gauge fixed per output."""
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        # Always construct the disposable bias to preserve initialization draws.
        super().__init__(in_features, out_features, bias=True)
        basis = _mean_zero_basis(in_features)
        self.weight = nn.Parameter(self.weight.detach() @ basis)
        self.register_buffer("basis", basis, persistent=False)
        if not bias:
            self.bias = None

    @property
    def projected_weight(self) -> torch.Tensor:
        return self.weight @ self.basis.transpose(0, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.projected_weight, self.bias)
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.qkv.bias = None
=======
        self.qkv = MeanZeroInputLinear(d_model, 3 * d_model, bias=False)
>>>>>>> REPLACE

<<<<<<< SEARCH
        qkv = F.linear(x, self.qkv.weight, qkv_bias)
=======
        qkv = F.linear(x, self.qkv.projected_weight, qkv_bias)
>>>>>>> REPLACE

<<<<<<< SEARCH
class TripleAnchoredScaleLayerNorm(nn.Module):
    """LayerNorm with three scales fixed through the following linear-map gauge."""
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim - 3))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = F.pad(self.weight, (0, 3), value=1.0)
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
        self.ln1 = TripleAnchoredScaleLayerNorm(cfg.d_model)
=======
        self.ln1 = QuadrupleAnchoredScaleLayerNorm(cfg.d_model)
>>>>>>> REPLACE

<<<<<<< SEARCH
                module.weight.copy_(full_weight @ module.basis)
                module.bias.zero_()
=======
                module.weight.copy_(full_weight @ module.basis)
                if module.bias is not None:
                    module.bias.zero_()
>>>>>>> REPLACE