MECHANISM: Shared adaptive pre-attention scale with a two-way final-LayerNorm scale tie

HYPOTHESIS: Combining the qualified four-terminal lag tie and shared adaptive `ln1` scale with one tied pair of final-LayerNorm scales will produce a 1,251-parameter model while retaining at least 99% accuracy, because seven independent final scales remain and prior LayerNorm-scale reductions were substantially more tolerant than the failed fifth routing-logit tie.

INTENDED_EDIT: Adopt the verified 1,252-parameter design, then reconstruct the final LayerNorm from seven learned scales by sharing its last learned scale across the final two channels.

EVIDENCE: The four-terminal-tie plus shared adaptive `ln1` design achieved 99.85% with 1,252 parameters, while fixing all `ln2` scales achieved 99.94% and seven-coordinate `ln1` quotienting achieved 99.96%; these results motivate removing one non-routing scale degree of freedom instead of extending the terminal lag tie that collapsed to 91.63%.

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
    """Final LayerNorm with its last two channel scales tied."""

    def __init__(self, d_model: int):
        super().__init__(d_model)
        self.weight = nn.Parameter(torch.ones(d_model - 1))

    def full_weight(self) -> torch.Tensor:
        return torch.cat((self.weight, self.weight[-1:]))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.layer_norm(
            x,
            self.normalized_shape,
            self.full_weight(),
            self.bias,
            self.eps,
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
        # Share one discrete kernel at a common temperature, fix the head
        # spacing, and tie the two terminal kernel logits.
        self.relative_bias = nn.Parameter(
            torch.zeros(max_seq_len - 2)
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
                self.relative_bias.new_zeros(2),
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