MECHANISM: Independent per-head query-key orthogonal gauge fixing

HYPOTHESIS: Extending the qualified 1537-parameter design with an independent query-key rotation in the second attention head will produce a 1536-parameter model with at least 99% accuracy, because the rotation preserves attention scores while fixing one additional query coefficient.

INTENDED_EDIT: Reproduce the qualified single-row QKV input gauge and first-head rotation, then rotate the second head’s first two query/key channels and omit its fixed query coefficient.

EVIDENCE: The single-row plus first-head rotation design achieved 99.96% accuracy at 1537 parameters; unlike the failed second mean-zero QKV-row constraint, this additional reduction uses a disjoint exact query-key symmetry.

<<<<<<< SEARCH
class MeanZeroOutputLinear(nn.Linear):
=======
class TwoHeadGaugeFixedQKV(nn.Module):
    """QKV map with one input-null and two query-key rotation gauges fixed."""
    def __init__(self, in_features: int, out_features: int, head_dim: int):
        super().__init__()
        if out_features != 3 * in_features:
            raise ValueError("gauge-fixed QKV requires out_features == 3 * in_features")
        if head_dim < 2 or 2 * head_dim > in_features:
            raise ValueError("two-head gauge fixing requires two nontrivial heads")
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
        self.leading_weight = nn.Parameter(fixed_weight[1:self.second_query])
        self.second_weight = nn.Parameter(
            fixed_weight[self.second_query, 1:]
        )
        self.trailing_weight = nn.Parameter(
            fixed_weight[self.second_query + 1:-1]
        )
        self.last_weight = nn.Parameter(fixed_weight[-1] @ basis)

    def _gauge_fix(self, full_weight: torch.Tensor) -> torch.Tensor:
        fixed_weight = full_weight.clone()
        for query_start in (0, self.second_query):
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
            self.leading_weight.copy_(
                fixed_weight[1:self.second_query]
            )
            self.second_weight.copy_(
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
        last_row = self.basis @ self.last_weight
        return torch.cat(
            (
                first_row.unsqueeze(0),
                self.leading_weight,
                second_row.unsqueeze(0),
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
        self.q_bias = nn.Parameter(torch.zeros(d_model))
=======
        self.qkv = TwoHeadGaugeFixedQKV(
            d_model, 3 * d_model, self.head_dim
        )
        self.q_bias = nn.Parameter(torch.zeros(d_model))
>>>>>>> REPLACE

<<<<<<< SEARCH
        elif isinstance(module, MeanZeroInputLinear):
            full_weight = module.weight.new_empty(
                module.out_features, module.in_features
            )
=======
        elif isinstance(module, TwoHeadGaugeFixedQKV):
            full_weight = module.leading_weight.new_empty(
                module.out_features, module.in_features
            )
            nn.init.normal_(full_weight, mean=0.0, std=0.02)
            module.reset_from_full(full_weight)
        elif isinstance(module, MeanZeroInputLinear):
            full_weight = module.weight.new_empty(
                module.out_features, module.in_features
            )
>>>>>>> REPLACE