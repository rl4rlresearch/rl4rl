MECHANISM: Eleven-unit learned nonlinear bottleneck

HYPOTHESIS: Starting from the qualified fully tied final-bias design, reducing the GELU MLP from 12 to 11 units will lower the model from 1,124 to 1,109 parameters while retaining at least 99% accuracy; the old designs assumed all 12 nonlinear units were load-bearing, while this tests a narrower learned representation without constraining the sensitive MLP output bias, attention bias, value maps, routing kernel, or adaptive scale.

INTENDED_EDIT: Adopt the verified zero-bias final LayerNorm and reduce `d_ff` by one, removing 15 learned MLP parameters while leaving training, checkpoints, causal attention, and decoding unchanged.

EVIDENCE: The fully tied final-bias model reached 99.84%–99.85% with 1,124 parameters. MLP output-bias tying collapsed to 91.52%, so this patch preserves every residual-bias coordinate and instead tests the previously unchallenged assumption that the pointwise computation requires a 12-unit hidden bank.

<<<<<<< SEARCH
class TiedFinalScaleLayerNorm(nn.LayerNorm):
    """Final LayerNorm with shared scale and five-way-tied zero-sum bias."""

    def __init__(self, d_model: int):
        super().__init__(d_model)
        self.weight = nn.Parameter(torch.ones(1))
        self.bias = nn.Parameter(torch.zeros(d_model - 5))

    def full_weight(self) -> torch.Tensor:
        return self.weight.expand(self.normalized_shape)

    def full_bias(self) -> torch.Tensor:
        anchored = torch.cat((self.bias, self.bias.new_zeros(5)))
        return anchored - anchored.mean()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.layer_norm(
            x,
            self.normalized_shape,
            self.full_weight(),
            self.full_bias(),
            self.eps,
        )
=======
class TiedFinalScaleLayerNorm(nn.LayerNorm):
    """Final LayerNorm with shared scale and fully tied zero-sum bias."""

    def __init__(self, d_model: int):
        super().__init__(d_model)
        self.weight = nn.Parameter(torch.ones(1))
        self.bias = None

    def full_weight(self) -> torch.Tensor:
        return self.weight.expand(self.normalized_shape)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.layer_norm(
            x,
            self.normalized_shape,
            self.full_weight(),
            None,
            self.eps,
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
    final_bias_gauge_parameters = [
        model.ln_f.bias
    ]
    last_coordinate_gauge_parameters = [
        *embedding_gauge_parameters,
        *output_bias_gauge_parameters,
        *attention_output_bias_gauge_parameters,
        *relative_bias_gauge_parameters,
        *final_bias_gauge_parameters,
    ]
=======
    last_coordinate_gauge_parameters = [
        *embedding_gauge_parameters,
        *output_bias_gauge_parameters,
        *attention_output_bias_gauge_parameters,
        *relative_bias_gauge_parameters,
    ]
>>>>>>> REPLACE

<<<<<<< SEARCH
    p.add_argument("--d-ff", type=int, default=12)
=======
    p.add_argument("--d-ff", type=int, default=11)
>>>>>>> REPLACE