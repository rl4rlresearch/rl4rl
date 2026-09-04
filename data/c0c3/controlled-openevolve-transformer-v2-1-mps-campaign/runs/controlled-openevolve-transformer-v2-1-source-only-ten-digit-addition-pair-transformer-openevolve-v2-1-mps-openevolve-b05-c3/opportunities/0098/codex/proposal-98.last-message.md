MECHANISM: Shared adaptive pre-attention scale

HYPOTHESIS: Reusing the final LayerNorm’s learned mean scale for the sole pre-attention LayerNorm scale will reduce the qualified four-terminal-tie model from 1,253 to 1,252 parameters while retaining at least 99% accuracy, because it preserves adaptive scale conditioning that was lost in the failed fully fixed `ln1` design.

INTENDED_EDIT: Adopt the verified four-coordinate terminal lag tie and replace the separate one-parameter `ln1` scale with a differentiable scale shared from the existing final LayerNorm weights.

EVIDENCE: Four terminal lag ties achieved 99.91% at 1,253 parameters, while five ties collapsed to 91.63%; separately, fixing the last adaptive `ln1` scale reached only 97.71%, motivating a non-routing reduction that preserves learned scale adaptation through parameter sharing.

<<<<<<< SEARCH
class GaugeFixedScaleLayerNorm(nn.LayerNorm):
    """LayerNorm with seven scales absorbed into the following value map."""

    def __init__(self, d_model: int):
        super().__init__(d_model)
        self.weight = nn.Parameter(torch.ones(d_model - 7))
        self.bias = None

    def full_weight(self) -> torch.Tensor:
        return torch.cat((self.weight, self.weight.new_ones(7)))
=======
class GaugeFixedScaleLayerNorm(nn.LayerNorm):
    """LayerNorm whose sole adaptive scale is shared with final LayerNorm."""

    def __init__(self, d_model: int):
        super().__init__(d_model)
        self.weight = None
        self.bias = None
        self.register_buffer("fixed_weight", torch.ones(d_model - 1))
        object.__setattr__(self, "_scale_source", None)

    def share_scale(self, source: nn.LayerNorm) -> None:
        object.__setattr__(self, "_scale_source", source)

    def full_weight(self) -> torch.Tensor:
        shared_scale = self._scale_source.weight.mean().reshape(1)
        return torch.cat((shared_scale, self.fixed_weight))
>>>>>>> REPLACE

<<<<<<< SEARCH
        # Share one discrete kernel at a common temperature, fix the head
        # spacing, and tie the three terminal kernel logits.
        self.relative_bias = nn.Parameter(
            torch.zeros(max_seq_len - 3)
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
                self.relative_bias.new_zeros(3),
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
        self.ln_f = nn.LayerNorm(cfg.d_model)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        for block in self.blocks:
            block.ln1.share_scale(self.ln_f)

        # Weight tying with the reconstructed full input embedding.
>>>>>>> REPLACE