MECHANISM: Three-coordinate pre-MLP LayerNorm-scale quotient

HYPOTHESIS: Extending the verified two-coordinate `ln2` scale quotient to three coordinates will reduce the model from 1,284 to 1,283 learned parameters while retaining at least 99% accuracy, because `fc1` can absorb the additional fixed channel scale.

INTENDED_EDIT: Adopt the qualified `fc1` LayerNorm-nullspace quotient and `ln2` bias absorption, then represent `ln2` with five learned scales and three fixed unit scales while preserving full-space gauge-aware optimization.

EVIDENCE: The two-coordinate `ln2` quotient achieved 99.96% accuracy with 1,284 parameters, after the one-coordinate version achieved 99.88%; the analogous incremental `ln1` quotient remained above 99% through seven fixed scales, motivating the next single-coordinate test.

<<<<<<< SEARCH
class GaugeFixedValueLinear(nn.Linear):
=======
class GaugeFixedMLPScaleLayerNorm(nn.LayerNorm):
    """LayerNorm with three scales absorbed into the following MLP map."""

    def __init__(self, d_model: int):
        super().__init__(d_model)
        self.weight = nn.Parameter(torch.ones(d_model - 3))
        self.bias = None

    def full_weight(self) -> torch.Tensor:
        return torch.cat((self.weight, self.weight.new_ones(3)))

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
class MLP(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc1.bias = nn.Parameter(torch.empty(d_ff - 1))
        self.fc2 = GaugeFixedMLPProjectionLinear(d_model, d_ff)
        self.fc2.bias = nn.Parameter(torch.empty(d_model - 1))
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bias = torch.cat((self.fc1.bias, self.fc1.bias.new_zeros(1)))
        hidden = F.linear(x, self.fc1.weight, bias)
        output_bias = torch.cat(
            (self.fc2.bias, self.fc2.bias.new_zeros(1))
        )
        output = F.linear(
            F.gelu(hidden), self.fc2.full_weight(), output_bias
        )
        return self.drop(output)
=======
class MLP(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float):
        super().__init__()
        self.fc1 = GaugeFixedFC1Linear(d_model, d_ff)
        self.fc2 = GaugeFixedMLPProjectionLinear(d_model, d_ff)
        self.fc2.bias = nn.Parameter(torch.empty(d_model - 1))
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden = self.fc1(x)
        output_bias = torch.cat(
            (self.fc2.bias, self.fc2.bias.new_zeros(1))
        )
        output = F.linear(
            F.gelu(hidden), self.fc2.full_weight(), output_bias
        )
        return self.drop(output)
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.ln2 = nn.LayerNorm(cfg.d_model)
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

<<<<<<< SEARCH
        for key_optimizer in key_optimizers:
            key_optimizer.step()
        for projection_optimizer in projection_optimizers:
            projection_optimizer.step()
        for mlp_projection_optimizer in mlp_projection_optimizers:
            mlp_projection_optimizer.step()
        for shared_optimizer in shared_optimizers:
            shared_optimizer.step()
        for gauge_optimizer in last_coordinate_optimizers:
            gauge_optimizer.step()
        optimizer.step()
        for value_optimizer in value_optimizers:
            value_optimizer.step()
=======
        for key_optimizer in key_optimizers:
            key_optimizer.step()
        for value_optimizer in value_optimizers:
            value_optimizer.step()
        for projection_optimizer in projection_optimizers:
            projection_optimizer.step()
        for mlp_projection_optimizer in mlp_projection_optimizers:
            mlp_projection_optimizer.step()
        for shared_optimizer in shared_optimizers:
            shared_optimizer.step()
        for gauge_optimizer in last_coordinate_optimizers:
            gauge_optimizer.step()
        optimizer.step()
>>>>>>> REPLACE