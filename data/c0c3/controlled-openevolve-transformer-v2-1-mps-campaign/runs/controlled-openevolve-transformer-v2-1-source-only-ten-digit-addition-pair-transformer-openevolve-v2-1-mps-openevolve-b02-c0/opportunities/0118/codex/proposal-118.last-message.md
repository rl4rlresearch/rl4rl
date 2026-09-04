MECHANISM: Nonadjacent attention-output common-mode gauge

HYPOTHESIS: Constraining the final attention-output projection column to zero mean will reduce the model to 1,574 parameters while retaining at least 99% accuracy, because this exact downstream-LayerNorm gauge avoids the fifth-column optimization sensitivity.

INTENDED_EDIT: Preserve the four verified leading attention-output gauges, additionally represent only the final column in the zero-mean basis, and initialize it by projecting an RNG-aligned full-column draw.

EVIDENCE: The analogous nonadjacent final-column gauge in `fc2` achieved 99.91% after its adjacent fourth-column extension failed; this directly motivates testing the final attention column after the adjacent fifth-column gauge failed at 70.06%.

<<<<<<< SEARCH
class AttentionGaugeLinear(nn.Linear):
    """Linear projection with zero-mean output bias and first four weight columns."""

    def __init__(self, d_model: int):
        super().__init__(d_model, d_model)
        basis = self.weight.detach().new_zeros(d_model, d_model - 1)
        for col in range(d_model - 1):
            scale = math.sqrt((col + 1) * (col + 2))
            basis[: col + 1, col] = 1.0 / scale
            basis[col + 1, col] = -(col + 1) / scale
        self.register_buffer("bias_basis", basis, persistent=False)

        full_weight = self.weight.detach()
        leading_weight_coords = basis.transpose(0, 1) @ full_weight[:, :4]
        self.weight = nn.Parameter(
            torch.cat((leading_weight_coords.flatten(), full_weight[:, 4:].flatten()))
        )
        self.bias = nn.Parameter(self.bias.detach()[:-1].clone())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gauge_size = 4 * (self.out_features - 1)
        leading_weight = self.bias_basis @ self.weight[:gauge_size].view(
            self.out_features - 1, 4
        )
        remaining_weight = self.weight[gauge_size:].view(
            self.out_features, self.in_features - 4
        )
        weight = torch.cat((leading_weight, remaining_weight), dim=1)
        return F.linear(x, weight, self.bias_basis @ self.bias)
=======
class AttentionGaugeLinear(nn.Linear):
    """Linear projection with zero-mean bias, first four columns, and final column."""

    def __init__(self, d_model: int):
        super().__init__(d_model, d_model)
        basis = self.weight.detach().new_zeros(d_model, d_model - 1)
        for col in range(d_model - 1):
            scale = math.sqrt((col + 1) * (col + 2))
            basis[: col + 1, col] = 1.0 / scale
            basis[col + 1, col] = -(col + 1) / scale
        self.register_buffer("bias_basis", basis, persistent=False)

        full_weight = self.weight.detach()
        leading_weight_coords = basis.transpose(0, 1) @ full_weight[:, :4]
        final_weight_coords = basis.transpose(0, 1) @ full_weight[:, -1:]
        self.weight = nn.Parameter(
            torch.cat(
                (
                    leading_weight_coords.flatten(),
                    full_weight[:, 4:-1].flatten(),
                    final_weight_coords.flatten(),
                )
            )
        )
        self.bias = nn.Parameter(self.bias.detach()[:-1].clone())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gauge_size = 4 * (self.out_features - 1)
        middle_end = gauge_size + self.out_features * (self.in_features - 5)
        leading_weight = self.bias_basis @ self.weight[:gauge_size].view(
            self.out_features - 1, 4
        )
        middle_weight = self.weight[gauge_size:middle_end].view(
            self.out_features, self.in_features - 5
        )
        final_weight = self.bias_basis @ self.weight[middle_end:].view(
            self.out_features - 1, 1
        )
        weight = torch.cat((leading_weight, middle_weight, final_weight), dim=1)
        return F.linear(x, weight, self.bias_basis @ self.bias)
>>>>>>> REPLACE

<<<<<<< SEARCH
    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, OutputAnchoredLinear):
=======
    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, AttentionGaugeLinear):
            gauge_size = 4 * (module.out_features - 1)
            baseline_weight = module.weight.new_empty(module.weight.numel() + 1)
            nn.init.normal_(baseline_weight, mean=0.0, std=0.02)
            baseline_leading = baseline_weight[:gauge_size]
            baseline_remaining = baseline_weight[gauge_size:].view(
                module.out_features, module.in_features - 4
            )
            final_weight_coords = (
                module.bias_basis.transpose(0, 1)
                @ baseline_remaining[:, -1:]
            )
            compact_weight = torch.cat(
                (
                    baseline_leading,
                    baseline_remaining[:, :-1].flatten(),
                    final_weight_coords.flatten(),
                )
            )
            with torch.no_grad():
                module.weight.copy_(compact_weight)
                module.bias.zero_()
        elif isinstance(module, OutputAnchoredLinear):
>>>>>>> REPLACE