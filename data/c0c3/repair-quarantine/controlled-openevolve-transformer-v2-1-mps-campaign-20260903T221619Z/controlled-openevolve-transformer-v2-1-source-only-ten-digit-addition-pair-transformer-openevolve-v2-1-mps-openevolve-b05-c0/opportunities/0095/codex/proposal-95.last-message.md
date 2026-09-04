MECHANISM: Common-output attention projection gauge

HYPOTHESIS: Gauge-fixing one attention output-weight coordinate will reduce the verified model from 1607 to 1606 parameters while retaining at least 99% accuracy, because a common feature shift in the attention output is removed by the following LayerNorms and the existing virtual AdamW machinery preserves the omitted coordinate’s optimization dynamics.

INTENDED_EDIT: Replace the attention output projection with a one-coordinate gauge-fixed projection and include it in the existing common-output virtual-gradient, AdamW, and clipping paths.

EVIDENCE: The verified 1607-parameter model achieved 99.97% accuracy with the same common-output weight quotient on the MLP projection. The attempted second MLP quotient produced no accuracy result because verification timed out, so applying the proven symmetry to the distinct attention output projection is an informative untested reduction.

<<<<<<< SEARCH
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.full_weight())


class CausalSelfAttention(nn.Module):
=======
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.full_weight())


class GaugeFixedAttentionOutput(nn.Module):
    """Attention output projection with one common-output coordinate fixed."""

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.fixed_weight_row = out_features // 2
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

    def forward(
        self,
        x: torch.Tensor,
        bias: torch.Tensor = None,
    ) -> torch.Tensor:
        return F.linear(x, self.full_weight(), bias)


class CausalSelfAttention(nn.Module):
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.q_bias = nn.Parameter(torch.zeros(d_model))
        self.virtual_v_bias_feature = 5
        self.proj = nn.Linear(d_model, d_model, bias=False)
        self.proj_bias = nn.Parameter(torch.zeros(d_model - 1))
=======
        self.q_bias = nn.Parameter(torch.zeros(d_model))
        self.virtual_v_bias_feature = 5
        self.proj = GaugeFixedAttentionOutput(d_model, d_model)
        self.proj_bias = nn.Parameter(torch.zeros(d_model - 1))
>>>>>>> REPLACE

<<<<<<< SEARCH
        y = F.linear(y, self.proj.weight, F.pad(self.proj_bias, (0, 1)))
=======
        y = self.proj(y, F.pad(self.proj_bias, (0, 1)))
>>>>>>> REPLACE

<<<<<<< SEARCH
        elif isinstance(module, GaugeFixedMLPOutput):
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
>>>>>>> REPLACE

<<<<<<< SEARCH
        for weight, bias, feature_index in self.gauges:
            if bias.grad is None:
                continue

            reduced_grad = bias.grad.detach().reshape(-1)
            full_grad = torch.cat(
                (
                    reduced_grad,
                    -reduced_grad.sum().reshape(1),
                )
            )
            virtual_grad = torch.dot(
                weight[:, feature_index].detach(),
                full_grad,
            )

            state = self.state[bias]
            state["step"] += 1
            step = state["step"]
            exp_avg = state["exp_avg"]
            exp_avg_sq = state["exp_avg_sq"]
            exp_avg.mul_(self.beta1).add_(
                virtual_grad,
                alpha=1.0 - self.beta1,
            )
            exp_avg_sq.mul_(self.beta2).addcmul_(
                virtual_grad,
                virtual_grad,
                value=1.0 - self.beta2,
            )

            bias_correction1 = 1.0 - self.beta1**step
            bias_correction2 = 1.0 - self.beta2**step
            direction = exp_avg / (
                exp_avg_sq.sqrt().div(
                    math.sqrt(bias_correction2)
                ).add(self.eps)
            )
            offset = -self.lr * direction / bias_correction1
            self.pending_offsets.append(
                (weight, bias, feature_index, offset)
            )
=======
        for projection, bias, feature_index in self.gauges:
            if bias.grad is None:
                continue

            reduced_grad = bias.grad.detach().reshape(-1)
            full_grad = torch.cat(
                (
                    reduced_grad,
                    -reduced_grad.sum().reshape(1),
                )
            )
            virtual_grad = torch.dot(
                projection.full_weight()[:, feature_index].detach(),
                full_grad,
            )

            state = self.state[bias]
            state["step"] += 1
            step = state["step"]
            exp_avg = state["exp_avg"]
            exp_avg_sq = state["exp_avg_sq"]
            exp_avg.mul_(self.beta1).add_(
                virtual_grad,
                alpha=1.0 - self.beta1,
            )
            exp_avg_sq.mul_(self.beta2).addcmul_(
                virtual_grad,
                virtual_grad,
                value=1.0 - self.beta2,
            )

            bias_correction1 = 1.0 - self.beta1**step
            bias_correction2 = 1.0 - self.beta2**step
            direction = exp_avg / (
                exp_avg_sq.sqrt().div(
                    math.sqrt(bias_correction2)
                ).add(self.eps)
            )
            offset = -self.lr * direction / bias_correction1
            self.pending_offsets.append(
                (projection, bias, feature_index, offset)
            )
>>>>>>> REPLACE

<<<<<<< SEARCH
    def project_biases(self) -> None:
        for weight, bias, feature_index, offset in self.pending_offsets:
            column = weight[:, feature_index]
            bias.add_((column[:-1] - column[-1]) * offset)
        self.pending_offsets = []
=======
    def project_biases(self) -> None:
        for projection, bias, feature_index, offset in self.pending_offsets:
            column = projection.full_weight()[:, feature_index]
            bias.add_((column[:-1] - column[-1]) * offset)
        self.pending_offsets = []
>>>>>>> REPLACE

<<<<<<< SEARCH
    # Recover the omitted value-bias gradient through the output projection.
    for weight, bias, feature_index in value_bias_gauges:
        if bias.grad is not None:
            reduced_grad = bias.grad.detach().reshape(-1).float()
            full_grad = torch.cat(
                (
                    reduced_grad,
                    -reduced_grad.sum().reshape(1),
                )
            )
            omitted = torch.dot(
                weight[:, feature_index].detach().float(),
                full_grad,
            )
            total_sq.add_(omitted.pow(2))
=======
    # Recover the omitted value-bias gradient through the output projection.
    for projection, bias, feature_index in value_bias_gauges:
        if bias.grad is not None:
            reduced_grad = bias.grad.detach().reshape(-1).float()
            full_grad = torch.cat(
                (
                    reduced_grad,
                    -reduced_grad.sum().reshape(1),
                )
            )
            omitted = torch.dot(
                projection.full_weight()[:, feature_index].detach().float(),
                full_grad,
            )
            total_sq.add_(omitted.pow(2))
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
        *[
            (
                block.mlp.fc2.weight,
                block.mlp.fc2.out_features,
                block.mlp.fc2.in_features,
                block.mlp.fc2.fixed_weight_row,
                block.mlp.fc2.fixed_weight_column,
            )
            for block in model.blocks
        ],
        *[
            (
                block.attn.proj.weight,
                block.attn.proj.out_features,
                block.attn.proj.in_features,
                block.attn.proj.fixed_weight_row,
                block.attn.proj.fixed_weight_column,
            )
            for block in model.blocks
        ],
    ]
>>>>>>> REPLACE

<<<<<<< SEARCH
    value_bias_gauges = [
        (
            block.attn.proj.weight,
            block.attn.proj_bias,
            block.attn.virtual_v_bias_feature,
        )
        for block in model.blocks
    ]
=======
    value_bias_gauges = [
        (
            block.attn.proj,
            block.attn.proj_bias,
            block.attn.virtual_v_bias_feature,
        )
        for block in model.blocks
    ]
>>>>>>> REPLACE