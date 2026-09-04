MECHANISM: Ambient-Adam terminal MLP weight-column gauge

HYPOTHESIS: Removing the all-ones output component of one `fc2` weight column while reproducing its full eight-coordinate AdamW dynamics will yield 1,625 parameters and retain at least 99% accuracy.

INTENDED_EDIT: Gauge-fix one terminal MLP output-weight column, preserve its original full-space initialization, and optimize its seven stored differences alongside the successful positional and terminal-bias gauges.

EVIDENCE: The 1,626-parameter design achieved 99.95% using an ambient-Adam gauge for the terminal MLP bias. A common offset across an `fc2` weight column likewise produces only an input-dependent all-ones residual immediately erased by the final LayerNorm, making this the closest untested extension of the successful bias gauge.

<<<<<<< SEARCH
class GaugeFixedBiasLinear(nn.Module):
    """Linear layer whose output bias omits its all-ones gauge scalar."""

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.empty(out_features - 1))
        self.full_bias = None
        self.reset_parameters()

    @torch.no_grad()
    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
        raw_bias = self.bias.new_empty(self.out_features)
        nn.init.uniform_(raw_bias, -bound, bound)
        self.bias.copy_(raw_bias[:-1] - raw_bias[-1])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        full_bias = torch.cat((self.bias, self.bias.new_zeros(1)))
        if torch.is_grad_enabled():
            full_bias.retain_grad()
            self.full_bias = full_bias
        return F.linear(x, self.weight, full_bias)
=======
class GaugeFixedBiasLinear(nn.Module):
    """Linear layer with gauges removed from its bias and one weight column."""

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight_column = nn.Parameter(torch.empty(out_features - 1))
        self.weight_rest = nn.Parameter(
            torch.empty(out_features, in_features - 1)
        )
        self.bias = nn.Parameter(torch.empty(out_features - 1))
        self.full_weight = None
        self.full_bias = None
        self.reset_parameters()

    @torch.no_grad()
    def reset_parameters(self) -> None:
        raw_weight = self.weight_rest.new_empty(
            self.out_features, self.in_features
        )
        nn.init.kaiming_uniform_(raw_weight, a=math.sqrt(5))
        self.weight_column.copy_(
            raw_weight[:-1, 0] - raw_weight[-1, 0]
        )
        self.weight_rest.copy_(raw_weight[:, 1:])
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(raw_weight)
        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
        raw_bias = self.bias.new_empty(self.out_features)
        nn.init.uniform_(raw_bias, -bound, bound)
        self.bias.copy_(raw_bias[:-1] - raw_bias[-1])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight_column = torch.cat(
            (self.weight_column, self.weight_column.new_zeros(1))
        )
        full_weight = torch.cat(
            (weight_column.unsqueeze(1), self.weight_rest), dim=1
        )
        full_bias = torch.cat((self.bias, self.bias.new_zeros(1)))
        if torch.is_grad_enabled():
            full_weight.retain_grad()
            full_bias.retain_grad()
            self.full_weight = full_weight
            self.full_bias = full_bias
        return F.linear(x, full_weight, full_bias)
>>>>>>> REPLACE

<<<<<<< SEARCH
        elif isinstance(module, GaugeFixedBiasLinear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            nn.init.zeros_(module.bias)
=======
        elif isinstance(module, GaugeFixedBiasLinear):
            raw_weight = module.weight_rest.new_empty(
                module.out_features, module.in_features
            )
            nn.init.normal_(raw_weight, mean=0.0, std=0.02)
            module.weight_column.data.copy_(
                raw_weight[:-1, 0] - raw_weight[-1, 0]
            )
            module.weight_rest.data.copy_(raw_weight[:, 1:])
            nn.init.zeros_(module.bias)
>>>>>>> REPLACE

<<<<<<< SEARCH
    # Optimize each seven-coordinate gauge parameter through the AdamW
    # moments of its corresponding full eight-coordinate vector.
    gauge_params = [model.pos_emb.first] + [
        blk.mlp.fc2.bias for blk in model.blocks
    ]
=======
    # Optimize each seven-coordinate gauge parameter through the AdamW
    # moments of its corresponding full eight-coordinate vector.
    gauge_params = (
        [model.pos_emb.first]
        + [blk.mlp.fc2.bias for blk in model.blocks]
        + [blk.mlp.fc2.weight_column for blk in model.blocks]
    )
>>>>>>> REPLACE

<<<<<<< SEARCH
        full_gauge_grads = [model.pos_emb.full_first.grad.detach()] + [
            blk.mlp.fc2.full_bias.grad.detach() for blk in model.blocks
        ]
=======
        full_gauge_grads = (
            [model.pos_emb.full_first.grad.detach()]
            + [
                blk.mlp.fc2.full_bias.grad.detach()
                for blk in model.blocks
            ]
            + [
                blk.mlp.fc2.full_weight.grad[:, 0].detach()
                for blk in model.blocks
            ]
        )
>>>>>>> REPLACE