MECHANISM: Virtual-AdamW quotient of an attention common-output weight direction

HYPOTHESIS: Gauge-fixing `attn.proj.weight[4, 7]` will reduce the verified model from 1610 to 1609 parameters while retaining at least 99% accuracy, because a common shift across the projection’s output features is removed by subsequent LayerNorms.

INTENDED_EDIT: Replace the attention output projection with a one-coordinate gauge-fixed projection and train its omitted coordinate using the existing virtual-gradient, AdamW-moment, and gauge-aware clipping machinery.

EVIDENCE: The analogous single-coordinate MLP output-weight quotient achieved 99.96% at 1610 parameters, while adding a second quotient within that same MLP matrix failed at 72.97%; testing the same exact null direction in the distinct attention projection is therefore the most informative next reduction.

<<<<<<< SEARCH
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.full_weight())


class CausalSelfAttention(nn.Module):
=======
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.full_weight())


class GaugeFixedAttentionOutput(nn.Module):
    """Attention output projection with one common-output direction removed."""

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.fixed_weight_row = 4
        self.fixed_weight_column = in_features - 1
        self.fixed_weight_index = (
            self.fixed_weight_row * in_features + self.fixed_weight_column
        )

        # Match nn.Linear's constructor-time random-number consumption.
        source = nn.Linear(in_features, out_features, bias=False)
        self.weight = nn.Parameter(
            self._reduce_weight(source.weight.detach())
        )

    def _weight_keep_mask(self, device: torch.device) -> torch.Tensor:
        keep = torch.ones(
            self.out_features * self.in_features,
            dtype=torch.bool,
            device=device,
        )
        keep[self.fixed_weight_index] = False
        return keep

    def _reduce_weight(self, full_weight: torch.Tensor) -> torch.Tensor:
        gauged = full_weight.clone()
        anchor = gauged[
            self.fixed_weight_row,
            self.fixed_weight_column,
        ].clone()
        gauged[:, self.fixed_weight_column].sub_(anchor)
        flat = gauged.reshape(-1)
        return flat[self._weight_keep_mask(flat.device)].clone()

    def full_weight(self) -> torch.Tensor:
        keep = self._weight_keep_mask(self.weight.device)
        flat = self.weight.new_zeros(keep.numel())
        flat = flat.masked_scatter(keep, self.weight)
        return flat.view(self.out_features, self.in_features)

    @torch.no_grad()
    def reset_weight_from_full_(self, full_weight: torch.Tensor) -> None:
        self.weight.copy_(self._reduce_weight(full_weight))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.full_weight())


class CausalSelfAttention(nn.Module):
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.v_bias = nn.Parameter(torch.zeros(d_model - 7))
        self.proj = nn.Linear(d_model, d_model, bias=False)
        self.proj_bias = nn.Parameter(torch.zeros(d_model - 1))
=======
        self.v_bias = nn.Parameter(torch.zeros(d_model - 7))
        self.proj = GaugeFixedAttentionOutput(d_model, d_model)
        self.proj_bias = nn.Parameter(torch.zeros(d_model - 1))
>>>>>>> REPLACE

<<<<<<< SEARCH
        y = F.linear(y, self.proj.weight, F.pad(self.proj_bias, (0, 1)))
=======
        y = F.linear(y, self.proj.full_weight(), F.pad(self.proj_bias, (0, 1)))
>>>>>>> REPLACE

<<<<<<< SEARCH
        elif isinstance(module, GaugeFixedMLPOutput):
            full_weight = torch.empty(
                module.out_features,
                module.in_features,
                device=module.weight.device,
                dtype=module.weight.dtype,
            )
            nn.init.normal_(full_weight, mean=0.0, std=0.02)
            module.reset_weight_from_full_(full_weight)
            nn.init.zeros_(module.bias)
=======
        elif isinstance(module, GaugeFixedAttentionOutput):
            full_weight = torch.empty(
                module.out_features,
                module.in_features,
                device=module.weight.device,
                dtype=module.weight.dtype,
            )
            nn.init.normal_(full_weight, mean=0.0, std=0.02)
            module.reset_weight_from_full_(full_weight)
        elif isinstance(module, GaugeFixedMLPOutput):
            full_weight = torch.empty(
                module.out_features,
                module.in_features,
                device=module.weight.device,
                dtype=module.weight.dtype,
            )
            nn.init.normal_(full_weight, mean=0.0, std=0.02)
            module.reset_weight_from_full_(full_weight)
            nn.init.zeros_(module.bias)
>>>>>>> REPLACE

<<<<<<< SEARCH
    output_weight_gauges = [
        (
            block.mlp.fc2.weight,
            block.mlp.fc2.out_features,
            block.mlp.fc2.in_features,
            block.mlp.fc2.fixed_weight_row,
            block.mlp.fc2.fixed_weight_column,
        )
        for block in model.blocks
    ]
=======
    output_weight_gauges = [
        (
            projection.weight,
            projection.out_features,
            projection.in_features,
            projection.fixed_weight_row,
            projection.fixed_weight_column,
        )
        for block in model.blocks
        for projection in (block.attn.proj, block.mlp.fc2)
    ]
>>>>>>> REPLACE