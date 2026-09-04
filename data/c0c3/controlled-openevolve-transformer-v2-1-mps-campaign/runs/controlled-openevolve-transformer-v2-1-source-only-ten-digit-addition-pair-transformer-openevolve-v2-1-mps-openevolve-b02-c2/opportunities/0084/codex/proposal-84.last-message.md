MECHANISM: Bias-absorbed LayerNorm query-row gauge quotient

HYPOTHESIS: Extending the verified 1,604-parameter design by Helmert-parameterizing one second-head query row will yield 1,603 parameters and at least 99% accuracy, because that row’s removed LayerNorm-uniform component is a position-independent query offset absorbable by its independent learned query bias.

INTENDED_EDIT: Reproduce the qualified two-coordinate `ln1.bias`, three-coordinate `ln2.bias`, and four-key-row compactions, then compact the first query row of the second attention head.

EVIDENCE: Four compact key rows achieved 99.87% at 1,604 parameters, while a fifth key row failed at 72.64%; the already-qualified biased `fc1` row demonstrates the independent LayerNorm affine-row quotient used here and motivates testing it on a query row instead of extending the failed axes.

<<<<<<< SEARCH
class CompactQKV(nn.Module):
    """Compact QKV with three LayerNorm-induced key-weight gauges fixed."""
=======
class CompactQKV(nn.Module):
    """Compact QKV with four key-row and one biased query-row gauges fixed."""
>>>>>>> REPLACE

<<<<<<< SEARCH
        retained_weight = torch.cat(
            (
                linear.weight[:key_start],
                linear.weight[key_start + 2 : second_head_start],
                linear.weight[second_head_start + 1 :],
            ),
            dim=0,
        )
=======
        retained_weight = torch.cat(
            (
                linear.weight[: self.head_dim],
                linear.weight[self.head_dim + 1 : key_start],
                linear.weight[key_start + 2 : second_head_start],
                linear.weight[second_head_start + 2 :],
            ),
            dim=0,
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
        scaled_key_weight = (
            linear.weight[[key_start, key_start + 1, second_head_start]]
            * ln_weight
        )
        centered_key_weight = (
            scaled_key_weight - scaled_key_weight.mean(dim=1, keepdim=True)
        )
        self.key_weight = nn.Parameter(
            (centered_key_weight @ basis).detach().clone()
        )
=======
        scaled_key_weight = (
            linear.weight[
                [
                    key_start,
                    key_start + 1,
                    second_head_start,
                    second_head_start + 1,
                ]
            ]
            * ln_weight
        )
        centered_key_weight = (
            scaled_key_weight - scaled_key_weight.mean(dim=1, keepdim=True)
        )
        self.key_weight = nn.Parameter(
            (centered_key_weight @ basis).detach().clone()
        )

        scaled_query_weight = linear.weight[self.head_dim] * ln_weight
        centered_query_weight = (
            scaled_query_weight - scaled_query_weight.mean()
        )
        self.query_weight = nn.Parameter(
            (basis.transpose(0, 1) @ centered_query_weight).detach().clone()
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scaled_key_weight = self.key_weight @ self.key_basis.transpose(0, 1)
        key_weight = scaled_key_weight / self.ln_weight
        second_retained_start = self.second_key_row - 2
        full_weight = torch.cat(
            (
                self.weight[: self.key_start],
                key_weight[:2],
                self.weight[self.key_start : second_retained_start],
                key_weight[2:],
                self.weight[second_retained_start:],
            ),
            dim=0,
        )
=======
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scaled_query_weight = self.key_basis @ self.query_weight
        query_weight = scaled_query_weight / self.ln_weight
        scaled_key_weight = self.key_weight @ self.key_basis.transpose(0, 1)
        key_weight = scaled_key_weight / self.ln_weight
        first_key_retained_start = self.key_start - 1
        second_key_retained_start = self.second_key_row - 3
        full_weight = torch.cat(
            (
                self.weight[: self.head_dim],
                query_weight.unsqueeze(0),
                self.weight[
                    self.head_dim : first_key_retained_start
                ],
                key_weight[:2],
                self.weight[
                    first_key_retained_start : second_key_retained_start
                ],
                key_weight[2:],
                self.weight[second_key_retained_start:],
            ),
            dim=0,
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
class CompactLayerNormBias(nn.Module):
    """LayerNorm with one downstream-linear-absorbed bias coordinate fixed."""

    def __init__(self, layer_norm: nn.LayerNorm):
        super().__init__()
        self.normalized_shape = layer_norm.normalized_shape
        self.eps = layer_norm.eps
        self.weight = layer_norm.weight
        self.bias = nn.Parameter(layer_norm.bias[:-1].detach().clone())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        full_bias = torch.cat((self.bias, self.bias.new_zeros(1)))
        return F.layer_norm(
            x,
            self.normalized_shape,
            self.weight,
            full_bias,
            self.eps,
        )
=======
class CompactLayerNormBias(nn.Module):
    """LayerNorm with downstream-linear-absorbed bias coordinates fixed."""

    def __init__(self, layer_norm: nn.LayerNorm, fixed_coordinates: int):
        super().__init__()
        self.normalized_shape = layer_norm.normalized_shape
        self.eps = layer_norm.eps
        self.fixed_coordinates = fixed_coordinates
        self.weight = layer_norm.weight
        self.bias = nn.Parameter(
            layer_norm.bias[:-fixed_coordinates].detach().clone()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        full_bias = torch.cat(
            (self.bias, self.bias.new_zeros(self.fixed_coordinates))
        )
        return F.layer_norm(
            x,
            self.normalized_shape,
            self.weight,
            full_bias,
            self.eps,
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.attn = CausalSelfAttention(cfg.d_model, cfg.n_head, cfg.dropout, cfg.max_seq_len)
        self.ln2 = CompactLayerNormBias(nn.LayerNorm(cfg.d_model))
=======
        self.ln1 = CompactLayerNormBias(
            nn.LayerNorm(cfg.d_model),
            fixed_coordinates=2,
        )
        self.attn = CausalSelfAttention(cfg.d_model, cfg.n_head, cfg.dropout, cfg.max_seq_len)
        self.ln2 = CompactLayerNormBias(
            nn.LayerNorm(cfg.d_model),
            fixed_coordinates=3,
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
        # Retain the qualified bias and projection layout, fix three key-row
        # gauges, and quotient one independently biased MLP input row.
=======
        # Retain the qualified bias and projection layout, fix four key-row
        # gauges plus one biased query-row gauge, and quotient one independently
        # biased MLP input row.
>>>>>>> REPLACE