MECHANISM: Two-coordinate pre-MLP LayerNorm-scale quotient

HYPOTHESIS: Extending the verified one-coordinate `ln2` scale quotient to two coordinates will reduce the qualified 1,285-parameter model to 1,284 parameters while retaining at least 99% accuracy, because `fc1` can absorb both fixed channel scales.

INTENDED_EDIT: Adopt the verified `fc1` LayerNorm-nullspace quotient, represent `ln2` with six learned scales and two fixed unit scales, and train both reduced input maps with the existing full-space gauge optimizer and clipping.

EVIDENCE: The 1,285-parameter design achieved 99.88% accuracy with one fixed `ln2` scale, while successive `ln1` quotients retained accuracy through seven fixed scales; a second incremental `ln2` coordinate is therefore the most direct boundary test.

<<<<<<< SEARCH
class GaugeFixedValueLinear(nn.Linear):
=======
class GaugeFixedMLPScaleLayerNorm(nn.LayerNorm):
    """LayerNorm with two scales absorbed into the following MLP map."""

    def __init__(self, d_model: int):
        super().__init__(d_model)
        self.weight = nn.Parameter(torch.ones(d_model - 2))
        self.bias = None

    def full_weight(self) -> torch.Tensor:
        return torch.cat((self.weight, self.weight.new_ones(2)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.layer_norm(
            x,
            self.normalized_shape,
            self.full_weight(),
            None,
            self.eps,
        )


class GaugeFixedFC1Linear(nn.Linear):
    """MLP input map modulo its LayerNorm-nullspace row shifts."""

    def __init__(self, d_model: int, d_ff: int):
        super().__init__(d_model, d_ff)
        self.d_model = d_model
        self.d_ff = d_ff
        self.weight = nn.Parameter(torch.empty(d_ff, d_model - 1))

    def full_weight(self) -> torch.Tensor:
        return torch.cat(
            (self.weight, self.weight.new_zeros(self.d_ff, 1)),
            dim=-1,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.full_weight(), self.bias)


class GaugeFixedValueLinear(nn.Linear):
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.fc1 = nn.Linear(d_model, d_ff)
=======
        self.fc1 = GaugeFixedFC1Linear(d_model, d_ff)
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.ln2 = nn.LayerNorm(cfg.d_model)
        self.ln2.bias = None
=======
        self.ln2 = GaugeFixedMLPScaleLayerNorm(cfg.d_model)
>>>>>>> REPLACE

<<<<<<< SEARCH
                module.weight.copy_(value[:, :-1])
        elif isinstance(module, GaugeTiedHead):
=======
                module.weight.copy_(value[:, :-1])
        elif isinstance(module, GaugeFixedFC1Linear):
            full = module.weight.new_empty(
                module.d_ff, module.d_model
            )
            nn.init.normal_(full, mean=0.0, std=0.02)
            with torch.no_grad():
                omitted = full[:, -1:].clone()
                full[:, :-1].sub_(omitted)
                full[:, -1].zero_()
                module.weight.copy_(full[:, :-1])
                nn.init.zeros_(module.bias)
        elif isinstance(module, GaugeTiedHead):
>>>>>>> REPLACE

<<<<<<< SEARCH
    value_gauge_modules = [
        (block.attn.value, block.ln1) for block in model.blocks
    ]
=======
    value_gauge_modules = [
        pair
        for block in model.blocks
        for pair in (
            (block.attn.value, block.ln1),
            (block.mlp.fc1, block.ln2),
        )
    ]
>>>>>>> REPLACE