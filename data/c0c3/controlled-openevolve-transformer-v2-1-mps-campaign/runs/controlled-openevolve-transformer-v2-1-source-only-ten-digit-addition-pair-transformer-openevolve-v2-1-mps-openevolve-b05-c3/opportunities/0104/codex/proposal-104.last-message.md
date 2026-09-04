MECHANISM: Six-way final-LayerNorm scale tying

HYPOTHESIS: Extending the qualified five-way final-scale tie to six channels will reduce the model from 1,248 to 1,247 learned parameters while retaining at least 99% accuracy, because three independent final scales and the shared adaptive pre-attention scale remain.

INTENDED_EDIT: Adopt the qualified four-terminal lag tie and shared adaptive `ln1` scale, then reconstruct the final LayerNorm from three learned scales with its last six channel scales tied.

EVIDENCE: The five-way final-scale tie achieved 99.83% accuracy with 1,248 parameters, after the preceding two-way through four-way ties also exceeded 99%; this motivates one further isolated scale tie while avoiding the fifth terminal lag tie that collapsed to 91.63%.

<<<<<<< SEARCH
class GaugeFixedScaleLayerNorm(nn.LayerNorm):
    """LayerNorm with seven scales absorbed into the following value map."""

    def __init__(self, d_model: int):
        super().__init__(d_model)
        self.weight = nn.Parameter(torch.ones(d_model - 7))
        self.bias = None

    def full_weight(self) -> torch.Tensor:
        return torch.cat((self.weight, self.weight.new_ones(7)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.layer_norm(
            x,
            self.normalized_shape,
            self.full_weight(),
            None,
            self.eps,
        )


class GaugeFixedMLPScaleLayerNorm(nn.LayerNorm):
=======
class GaugeFixedScaleLayerNorm(nn.LayerNorm):
    """LayerNorm whose sole adaptive scale is shared with final LayerNorm."""

    def __init__(self, d_model: int):
        super().__init__(d_model)
        self.weight = None
        self.bias = None
        self.register_buffer("fixed_weight", torch.ones(d_model - 1))
        object.__setattr__(self, "_scale_source", None)

    def share_scale(self, source: nn.Module) -> None:
        object.__setattr__(self, "_scale_source", source)

    def full_weight(self) -> torch.Tensor:
        source_weight = self._scale_source.full_weight()
        shared_scale = source_weight.mean().reshape(1)
        return torch.cat((shared_scale, self.fixed_weight))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.layer_norm(
            x,
            self.normalized_shape,
            self.full_weight(),
            None,
            self.eps,
        )


class TiedFinalScaleLayerNorm(nn.LayerNorm):
    """Final LayerNorm with its last six channel scales tied."""

    def __init__(self, d_model: int):
        super().__init__(d_model)
        self.weight = nn.Parameter(torch.ones(d_model - 5))

    def full_weight(self) -> torch.Tensor:
        return torch.cat((self.weight, self.weight[-1:].expand(5)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.layer_norm(
            x,
            self.normalized_shape,
            self.full_weight(),
            self.bias,
            self.eps,
        )


class GaugeFixedMLPScaleLayerNorm(nn.LayerNorm):
>>>>>>> REPLACE

<<<<<<< SEARCH
        # Share one unrestricted discrete kernel while allowing each head to
        # learn a cyclic displacement at a common softmax temperature.
        self.relative_bias = nn.Parameter(
            torch.zeros(max_seq_len - 1)
        )
=======
        # Share one discrete kernel at a common temperature, fix the head
        # spacing, and tie the four terminal kernel logits.
        self.relative_bias = nn.Parameter(
            torch.zeros(max_seq_len - 4)
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
        base_bias = torch.cat(
            (
                self.relative_bias,
                self.relative_bias.new_zeros(1),
            )
        )
=======
        base_bias = torch.cat(
            (
                self.relative_bias,
                self.relative_bias.new_zeros(4),
            )
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.ln_f = nn.LayerNorm(cfg.d_model)

        # Weight tying with the reconstructed full input embedding.
=======
        self.drop = nn.Dropout(cfg.dropout)
        self.ln_f = TiedFinalScaleLayerNorm(cfg.d_model)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        for block in self.blocks:
            block.ln1.share_scale(self.ln_f)

        # Weight tying with the reconstructed full input embedding.
>>>>>>> REPLACE