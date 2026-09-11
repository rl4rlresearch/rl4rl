MECHANISM: Second-coordinate pre-attention scale-gauge anchoring

HYPOTHESIS: Combining the qualified mean-zero `fc1` parameterization with two fixed `ln1` scale coordinates will produce a 1541-parameter model with at least 99% accuracy, because the second fixed scale can be absorbed into its unrestricted `qkv` input column while six learned scales remain.

INTENDED_EDIT: Reproduce the qualified `fc1` null-space reduction and extend the successful single-coordinate `ln1` scale anchor to the final two coordinates.

EVIDENCE: Mean-zero `fc1` achieved 99.83% at 1543 parameters, and fixing one `ln1` scale retained 99.76% at 1542; this supports testing one incremental scale anchor, while the 71.25% result from removing all eight scales cautions against a larger jump.

<<<<<<< SEARCH
class MeanZeroOutputLinear(nn.Linear):
    """Linear map with its residual-stream common mode gauge-fixed."""
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


class MeanZeroOutputLinear(nn.Linear):
    """Linear map with its residual-stream common mode gauge-fixed."""
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.fc1 = nn.Linear(d_model, d_ff)
=======
        self.fc1 = MeanZeroInputLinear(d_model, d_ff)
>>>>>>> REPLACE

<<<<<<< SEARCH
class Block(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.d_model, bias=False)
=======
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


class Block(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.ln1 = TwiceAnchoredScaleLayerNorm(cfg.d_model)
>>>>>>> REPLACE

<<<<<<< SEARCH
        elif isinstance(module, MeanZeroOutputLinear):
            full_weight = module.weight.new_empty(
                module.out_features, module.in_features
            )
=======
        elif isinstance(module, MeanZeroInputLinear):
            full_weight = module.weight.new_empty(
                module.out_features, module.in_features
            )
            nn.init.normal_(full_weight, mean=0.0, std=0.02)
            with torch.no_grad():
                module.weight.copy_(full_weight @ module.basis)
                module.bias.zero_()
        elif isinstance(module, MeanZeroOutputLinear):
            full_weight = module.weight.new_empty(
                module.out_features, module.in_features
            )
>>>>>>> REPLACE