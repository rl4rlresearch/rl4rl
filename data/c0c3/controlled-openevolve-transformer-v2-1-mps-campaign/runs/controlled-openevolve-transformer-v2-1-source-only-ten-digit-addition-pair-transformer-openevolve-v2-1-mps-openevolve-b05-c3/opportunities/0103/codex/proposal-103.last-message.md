MECHANISM: Five-way final-LayerNorm scale tying

HYPOTHESIS: Extending the qualified four-way final-scale tie to five channels will reduce the model from 1,249 to 1,248 learned parameters while retaining at least 99% accuracy, because four independent final scales and the shared adaptive pre-attention scale remain.

INTENDED_EDIT: Adopt the qualified shared adaptive `ln1` scale, reconstruct the final LayerNorm from four learned scales with its last five channel scales tied, and leave attention, training, checkpointing, and decoding unchanged.

EVIDENCE: The four-way final-scale tie achieved 99.85% accuracy with 1,249 parameters after the two-way and three-way ties also exceeded 99%, directly motivating one further isolated final-scale tie.

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
    """Final LayerNorm with its last five channel scales tied."""

    def __init__(self, d_model: int):
        super().__init__(d_model)
        self.weight = nn.Parameter(torch.ones(d_model - 4))

    def full_weight(self) -> torch.Tensor:
        return torch.cat((self.weight, self.weight[-1:].expand(4)))

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