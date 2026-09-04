MECHANISM: Third-coordinate pre-attention scale-gauge anchoring

HYPOTHESIS: Extending the qualified two-coordinate `ln1` scale anchor to three coordinates will produce a 1540-parameter model with at least 99% accuracy, because the third fixed scale can be absorbed into the corresponding unrestricted `qkv` input column while five learned scales remain.

INTENDED_EDIT: Reproduce the qualified mean-zero `fc1` and affine-free `ln2` reductions, then fix the final three `ln1` scale coordinates at one.

EVIDENCE: The two-coordinate anchor achieved 99.82% accuracy with 1541 parameters, supporting one incremental anchor; removing all eight `ln1` scales collapsed to 71.25%, so this retains most learned scales.

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
        return F.linear(x, self.weight @ self.basis.transpose(0, 1), self.bias)


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
        self.attn = CausalSelfAttention(cfg.d_model, cfg.n_head, cfg.dropout, cfg.max_seq_len)
        self.ln2 = nn.LayerNorm(cfg.d_model, bias=False)
        self.mlp = MLP(cfg.d_model, cfg.d_ff, cfg.dropout)
=======
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


class Block(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.ln1 = TripleAnchoredScaleLayerNorm(cfg.d_model)
        self.attn = CausalSelfAttention(cfg.d_model, cfg.n_head, cfg.dropout, cfg.max_seq_len)
        self.ln2 = nn.LayerNorm(cfg.d_model, elementwise_affine=False)
        self.mlp = MLP(cfg.d_model, cfg.d_ff, cfg.dropout)
>>>>>>> REPLACE

<<<<<<< SEARCH
        elif isinstance(module, MeanZeroOutputLinear):
            full_weight = module.weight.new_empty(
                module.out_features, module.in_features
            )
            nn.init.normal_(full_weight, mean=0.0, std=0.02)
            with torch.no_grad():
                module.weight.copy_(
                    module.basis.transpose(0, 1) @ full_weight
                )
                module.bias.zero_()
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
            nn.init.normal_(full_weight, mean=0.0, std=0.02)
            with torch.no_grad():
                module.weight.copy_(
                    module.basis.transpose(0, 1) @ full_weight
                )
                module.bias.zero_()
>>>>>>> REPLACE