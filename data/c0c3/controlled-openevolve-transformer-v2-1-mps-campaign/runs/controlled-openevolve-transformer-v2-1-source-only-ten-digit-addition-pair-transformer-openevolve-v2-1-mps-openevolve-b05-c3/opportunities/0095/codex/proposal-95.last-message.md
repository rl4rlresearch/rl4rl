MECHANISM: Three-way terminal lag-logit tying

HYPOTHESIS: Extending the qualified two-way terminal lag tie to three coordinates will reduce the model from 1,255 to 1,254 learned parameters while retaining at least 99% accuracy, because the unrestricted remaining lag kernel preserves nearly all routing capacity.

INTENDED_EDIT: Adopt fixed `ln2` scales, fixed half-period head spacing, and common head temperature, then tie the final three cyclic-kernel logits to the gauge-fixed value.

EVIDENCE: The 1,255-parameter design with two tied terminal lag logits achieved 99.83% accuracy; its 0.83-point margin above the requirement motivates the next isolated one-coordinate tie.

<<<<<<< SEARCH
class GaugeFixedMLPScaleLayerNorm(nn.LayerNorm):
    """LayerNorm with seven scales absorbed into the following MLP map."""

    def __init__(self, d_model: int):
        super().__init__(d_model)
        self.weight = nn.Parameter(torch.ones(d_model - 7))
        self.bias = None

    def full_weight(self) -> torch.Tensor:
        return torch.cat((self.weight, self.weight.new_ones(7)))
=======
class GaugeFixedMLPScaleLayerNorm(nn.LayerNorm):
    """LayerNorm with all scales absorbed into the following MLP map."""

    def __init__(self, d_model: int):
        super().__init__(d_model)
        self.weight = None
        self.bias = None
        self.register_buffer("fixed_weight", torch.ones(d_model))

    def full_weight(self) -> torch.Tensor:
        return self.fixed_weight
>>>>>>> REPLACE

<<<<<<< SEARCH
        # Share one unrestricted discrete kernel while allowing each head to
        # learn a cyclic displacement and softmax temperature.
        self.relative_bias = nn.Parameter(
            torch.zeros(max_seq_len - 1)
        )
        self.relative_shift = nn.Parameter(
            torch.arange(1, n_head, dtype=torch.float32)
            * (max_seq_len / n_head)
        )
        self.relative_log_scale = nn.Parameter(
            torch.zeros(n_head - 1)
        )
=======
        # Share one discrete kernel at a common temperature, fix the head
        # spacing, and tie the three terminal kernel logits.
        self.relative_bias = nn.Parameter(
            torch.zeros(max_seq_len - 3)
        )
        self.register_buffer(
            "relative_shift",
            torch.arange(1, n_head, dtype=torch.float32)
            * (max_seq_len / n_head),
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
                self.relative_bias.new_zeros(3),
            )
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
        head_log_scale = torch.cat(
            (
                self.relative_log_scale.new_zeros(1),
                self.relative_log_scale,
            )
        )
        phase = torch.exp(
=======
        phase = torch.exp(
>>>>>>> REPLACE

<<<<<<< SEARCH
        lag_bias = lag_bias * head_log_scale.exp()[:, None]
        lag_bias = lag_bias - lag_bias[:, -1:]
=======
        lag_bias = lag_bias - lag_bias[:, -1:]
>>>>>>> REPLACE