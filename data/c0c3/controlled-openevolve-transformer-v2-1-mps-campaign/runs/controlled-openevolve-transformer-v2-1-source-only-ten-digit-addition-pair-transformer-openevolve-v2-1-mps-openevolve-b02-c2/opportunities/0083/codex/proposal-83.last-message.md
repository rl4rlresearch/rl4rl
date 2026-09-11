MECHANISM: Fifth LayerNorm-induced key-row gauge quotient

HYPOTHESIS: Reproducing the verified 1,604-parameter design and compacting a fifth key-weight row will yield 1,603 parameters with at least 99% accuracy, because centering its LayerNorm-scaled weights changes that key coordinate only by a source-position-independent offset canceled by attention softmax.

INTENDED_EDIT: Apply the verified two-coordinate `ln1.bias` compaction and extend `CompactQKV` from three to five Helmert-parameterized key rows.

EVIDENCE: Four compact key rows achieved 99.87% accuracy at 1,604 parameters, while removing a third `ln1.bias` coordinate failed at 96.07%; extending the exact key-row invariance is therefore the strongest supported next reduction.

<<<<<<< SEARCH
class CompactQKV(nn.Module):
    """Compact QKV with three LayerNorm-induced key-weight gauges fixed."""
=======
class CompactQKV(nn.Module):
    """Compact QKV with five LayerNorm-induced key-weight gauges fixed."""
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
                linear.weight[:key_start],
                linear.weight[key_start + 3 : second_head_start],
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
=======
        scaled_key_weight = (
            linear.weight[
                [
                    key_start,
                    key_start + 1,
                    key_start + 2,
                    second_head_start,
                    second_head_start + 1,
                ]
            ]
            * ln_weight
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
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
        second_retained_start = self.second_key_row - 3
        full_weight = torch.cat(
            (
                self.weight[: self.key_start],
                key_weight[:3],
                self.weight[self.key_start : second_retained_start],
                key_weight[3:],
                self.weight[second_retained_start:],
            ),
            dim=0,
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
class CompactLayerNormBias(nn.Module):
    """LayerNorm with three downstream-linear-absorbed bias coordinates fixed."""

    def __init__(self, layer_norm: nn.LayerNorm):
        super().__init__()
        self.normalized_shape = layer_norm.normalized_shape
        self.eps = layer_norm.eps
        self.weight = layer_norm.weight
        self.bias = nn.Parameter(layer_norm.bias[:-3].detach().clone())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        full_bias = torch.cat((self.bias, self.bias.new_zeros(3)))
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
        # Retain the qualified bias and projection layout, fix five key-row
        # gauges, and quotient one independently biased MLP input row.
>>>>>>> REPLACE