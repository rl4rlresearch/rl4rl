MECHANISM: Additional query-key rotational gauge fixing

HYPOTHESIS: Adding a second sequential query-key rotation within the first attention head will produce a 1535-parameter model with at least 99% accuracy because it fixes one additional query coefficient while preserving attention scores.

INTENDED_EDIT: Reproduce the qualified 1536-parameter four-anchor, single-row QKV design and add one further orthogonal query-key rotation in the first head.

EVIDENCE: Independent rotations in the first and second heads achieved 99.57% accuracy at 1536 parameters; unlike the failed second mean-zero QKV-row constraint, this removes another exact rotational gauge direction.

<<<<<<< SEARCH
class MeanZeroOutputLinear(nn.Linear):
=======
class ThreeRotationGaugeFixedQKV(nn.Module):
    """QKV map with one input-null and three query-key rotations fixed."""
    def __init__(self, in_features: int, out_features: int, head_dim: int):
        super().__init__()
        if out_features != 3 * in_features:
            raise ValueError("gauge-fixed QKV requires out_features == 3 * in_features")
        if head_dim < 3 or 2 * head_dim > in_features:
            raise ValueError("three-rotation gauge fixing requires two nontrivial heads")
        self.in_features = in_features
        self.out_features = out_features
        self.second_query = head_dim
        basis = _mean_zero_basis(in_features)
        self.register_buffer("basis", basis, persistent=False)

        full_weight = torch.empty(out_features, in_features)
        nn.init.kaiming_uniform_(full_weight, a=math.sqrt(5))
        discarded_bias = torch.empty(out_features)
        bound = 1.0 / math.sqrt(in_features)
        nn.init.uniform_(discarded_bias, -bound, bound)

        fixed_weight = self._gauge_fix(full_weight)
        self.first_weight = nn.Parameter(fixed_weight[0, 1:])
        self.second_weight = nn.Parameter(fixed_weight[1, 1:])
        self.leading_weight = nn.Parameter(
            fixed_weight[2:self.second_query]
        )
        self.head_two_weight = nn.Parameter(
            fixed_weight[self.second_query, 1:]
        )
        self.trailing_weight = nn.Parameter(
            fixed_weight[self.second_query + 1:-1]
        )
        self.last_weight = nn.Parameter(fixed_weight[-1] @ basis)

    def _gauge_fix(self, full_weight: torch.Tensor) -> torch.Tensor:
        fixed_weight = full_weight.clone()
        for query_start in (0, 1, self.second_query):
            pivot = fixed_weight[query_start:query_start + 2, 0]
            radius = pivot.norm().clamp_min(
                torch.finfo(full_weight.dtype).tiny
            )
            cosine = pivot[1] / radius
            sine = -pivot[0] / radius
            rotation = torch.stack(
                (
                    torch.stack((cosine, sine)),
                    torch.stack((-sine, cosine)),
                )
            )
            fixed_weight[query_start:query_start + 2] = (
                rotation @ fixed_weight[query_start:query_start + 2]
            )
            key_start = self.in_features + query_start
            fixed_weight[key_start:key_start + 2] = (
                rotation @ fixed_weight[key_start:key_start + 2]
            )
            fixed_weight[query_start, 0] = 0.0
        return fixed_weight

    def reset_from_full(self, full_weight: torch.Tensor) -> None:
        fixed_weight = self._gauge_fix(full_weight)
        with torch.no_grad():
            self.first_weight.copy_(fixed_weight[0, 1:])
            self.second_weight.copy_(fixed_weight[1, 1:])
            self.leading_weight.copy_(
                fixed_weight[2:self.second_query]
            )
            self.head_two_weight.copy_(
                fixed_weight[self.second_query, 1:]
            )
            self.trailing_weight.copy_(
                fixed_weight[self.second_query + 1:-1]
            )
            self.last_weight.copy_(fixed_weight[-1] @ self.basis)

    @property
    def weight(self) -> torch.Tensor:
        first_row = F.pad(self.first_weight, (1, 0))
        second_row = F.pad(self.second_weight, (1, 0))
        head_two_row = F.pad(self.head_two_weight, (1, 0))
        last_row = self.basis @ self.last_weight
        return torch.cat(
            (
                first_row.unsqueeze(0),
                second_row.unsqueeze(0),
                self.leading_weight,
                head_two_row.unsqueeze(0),
                self.trailing_weight,
                last_row.unsqueeze(0),
            ),
            dim=0,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight)


class MeanZeroOutputLinear(nn.Linear):
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.qkv.bias = None
=======
        self.qkv = ThreeRotationGaugeFixedQKV(
            d_model, 3 * d_model, self.head_dim
        )
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
        elif isinstance(module, MeanZeroInputLinear):
=======
        elif isinstance(module, ThreeRotationGaugeFixedQKV):
            full_weight = module.leading_weight.new_empty(
                module.out_features, module.in_features
            )
            nn.init.normal_(full_weight, mean=0.0, std=0.02)
            module.reset_from_full(full_weight)
        elif isinstance(module, MeanZeroInputLinear):
>>>>>>> REPLACE