MECHANISM: Equal-temperature phase-shared lag attention

HYPOTHESIS: Fixing the relative head temperature in the verified phase-shared, fully scale-absorbed design will retain at least 99% accuracy while reducing the model from 1,258 to 1,257 learned parameters, because the learned cyclic phase still provides distinct head routing.

INTENDED_EDIT: Adopt phase-shared discrete lag attention and complete `ln2` scale absorption, while fixing both heads to the shared kernel temperature.

EVIDENCE: Phase-shared lag attention combined with complete `ln2` scale absorption achieved 99.98% accuracy with 1,258 parameters; removing only its single relative-temperature parameter isolates whether phase separation alone supplies the necessary head specialization.

<<<<<<< SEARCH
class GaugeFixedMLPScaleLayerNorm(nn.LayerNorm):
    """LayerNorm with five scales absorbed into the following MLP map."""

    def __init__(self, d_model: int):
        super().__init__(d_model)
        self.weight = nn.Parameter(torch.ones(d_model - 5))
        self.bias = None

    def full_weight(self) -> torch.Tensor:
        return torch.cat((self.weight, self.weight.new_ones(5)))
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
        # learn a cyclic displacement at a common softmax temperature.
        self.relative_bias = nn.Parameter(
            torch.zeros(max_seq_len - 1)
        )
        self.relative_shift = nn.Parameter(
            torch.arange(1, n_head, dtype=torch.float32)
            * (max_seq_len / n_head)
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
        lag_bias = lag_bias - lag_bias[:, -1:]
        att = lag_bias[:, lag].unsqueeze(0)
>>>>>>> REPLACE