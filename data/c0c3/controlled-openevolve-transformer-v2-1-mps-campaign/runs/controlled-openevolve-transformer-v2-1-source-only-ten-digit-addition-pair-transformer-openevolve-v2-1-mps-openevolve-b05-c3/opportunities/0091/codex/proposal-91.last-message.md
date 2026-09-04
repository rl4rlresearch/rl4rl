MECHANISM: Phase-shared lag attention with complete pre-MLP scale absorption

HYPOTHESIS: Combining the verified phase-shared lag kernel with complete `ln2` scale absorption will reduce the qualified 1,259-parameter design by one learned parameter while retaining at least 99% accuracy.

INTENDED_EDIT: Replace independent per-head lag tables with one learned full-resolution kernel plus a learned relative phase and temperature, and make every `ln2` scale fixed at one.

EVIDENCE: Phase-shared discrete lag attention achieved 99.95% accuracy with 1,259 parameters, while complete `ln2` scale absorption independently achieved 99.94%; this directly tests whether the two successful reductions compose.

<<<<<<< SEARCH
class GaugeFixedMLPScaleLayerNorm(nn.LayerNorm):
    """LayerNorm with six scales absorbed into the following MLP map."""

    def __init__(self, d_model: int):
        super().__init__(d_model)
        self.weight = nn.Parameter(torch.ones(d_model - 6))
        self.bias = None

    def full_weight(self) -> torch.Tensor:
        return torch.cat((self.weight, self.weight.new_ones(6)))
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
        # Each head learns a stationary causal routing preference. The omitted
        # final lag fixes the softmax-invariant common shift of each table.
        self.relative_bias = nn.Parameter(
            torch.zeros(n_head, max_seq_len - 1)
        )
=======
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
>>>>>>> REPLACE

<<<<<<< SEARCH
        lag_bias = torch.cat(
            (
                self.relative_bias,
                self.relative_bias.new_zeros(self.n_head, 1),
            ),
            dim=-1,
        )
        att = lag_bias[:, lag].unsqueeze(0)
=======
        base_bias = torch.cat(
            (
                self.relative_bias,
                self.relative_bias.new_zeros(1),
            )
        )
        frequency = torch.arange(
            base_bias.numel() // 2 + 1,
            device=x.device,
            dtype=base_bias.dtype,
        )
        head_shift = torch.cat(
            (
                self.relative_shift.new_zeros(1),
                self.relative_shift,
            )
        )
        head_log_scale = torch.cat(
            (
                self.relative_log_scale.new_zeros(1),
                self.relative_log_scale,
            )
        )
        phase = torch.exp(
            -2j
            * math.pi
            * head_shift[:, None]
            * frequency[None, :]
            / base_bias.numel()
        )
        lag_bias = torch.fft.irfft(
            torch.fft.rfft(base_bias).unsqueeze(0) * phase,
            n=base_bias.numel(),
            dim=-1,
        )
        lag_bias = lag_bias * head_log_scale.exp()[:, None]
        lag_bias = lag_bias - lag_bias[:, -1:]
        att = lag_bias[:, lag].unsqueeze(0)
>>>>>>> REPLACE