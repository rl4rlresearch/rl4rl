MECHANISM: Fourth query-key orthogonal gauge fixing

HYPOTHESIS: Adding a third sequential rotation within the first attention head will produce a 1534-parameter model with at least 99% accuracy because it removes another exact query-key rotational gauge while preserving attention scores.

INTENDED_EDIT: Reproduce the qualified four-scale anchor, single mean-zero QKV row, and three rotations, then rotate first-head query/key channels 2–3 and omit the resulting fixed query coefficient.

EVIDENCE: The three-rotation 1535-parameter design achieved 99.84% accuracy; extending its exact rotational symmetry is better supported than another mean-zero QKV-row constraint, whose second application fell to 96.45%.

<<<<<<< SEARCH
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight @ self.basis.transpose(0, 1), self.bias)


class MeanZeroOutputLinear(nn.Linear):
=======
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight @ self.basis.transpose(0, 1), self.bias)


class FourRotationGaugeFixedQKV(nn.Module):
    """QKV map with one input-null and four query-key rotations fixed."""
    def __init__(self, in_features: int, out_features: int, head_dim: int):
        super().__init__()
        if out_features != 3 * in_features:
            raise ValueError("gauge-fixed QKV requires out_features == 3 * in_features")
        if head_dim < 4 or 2 * head_dim > in_features:
            raise ValueError("four-rotation gauge fixing requires two suitable heads")
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
        self.third_weight = nn.Parameter(fixed_weight[2, 1:])
        self.first_head_tail = nn.Parameter(
            fixed_weight[3:self.second_query]
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
        for query_start in (0, 1, 2, self.second_query):
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
            self.third_weight.copy_(fixed_weight[2, 1:])
            self.first_head_tail.copy_(
                fixed_weight[3:self.second_query]
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
        third_row = F.pad(self.third_weight, (1, 0))
        head_two_row = F.pad(self.head_two_weight, (1, 0))
        last_row = self.basis @ self.last_weight
        return torch.cat(
            (
                first_row.unsqueeze(0),
                second_row.unsqueeze(0),
                third_row.unsqueeze(0),
                self.first_head_tail,
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
        self.n_head = n_head
        self.head_dim = d_model // n_head
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.qkv.bias = None
        self.q_bias = nn.Parameter(torch.zeros(d_model))
=======
        self.n_head = n_head
        self.head_dim = d_model // n_head
        self.qkv = FourRotationGaugeFixedQKV(
            d_model, 3 * d_model, self.head_dim
        )
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
=======
        elif isinstance(module, FourRotationGaugeFixedQKV):
            full_weight = module.first_weight.new_empty(
                module.out_features, module.in_features
            )
            nn.init.normal_(full_weight, mean=0.0, std=0.02)
            module.reset_from_full(full_weight)
        elif isinstance(module, MeanZeroInputLinear):
            full_weight = module.weight.new_empty(
                module.out_features, module.in_features
            )
>>>>>>> REPLACE