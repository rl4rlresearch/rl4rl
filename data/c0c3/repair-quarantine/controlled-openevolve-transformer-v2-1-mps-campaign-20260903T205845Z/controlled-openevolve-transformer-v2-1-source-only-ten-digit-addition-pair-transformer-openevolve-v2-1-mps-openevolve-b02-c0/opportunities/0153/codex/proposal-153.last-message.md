MECHANISM: RNG-preserving fifth attention-output gauge

HYPOTHESIS: Reducing the fifth attention-output column to seven zero-mean coordinates will achieve at least 99% accuracy when initialization consumes the same 60 random draws as the verified 1,576-parameter model, preserving all downstream initial weights.

INTENDED_EDIT: Extend the attention-output gauge from four to five columns and reconstruct an equivalent 59-parameter weight, while projecting the fifth column from a 60-draw baseline initialization.

EVIDENCE: The prior fifth-column gauge reached only 79.0%, but the supplied generic initializer draws according to the reduced parameter count and therefore shifts every downstream random initialization; the 1,576-parameter model reached 99.92%, motivating an initialization-controlled retry of the same exact residual common-mode invariance.

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
    """Linear projection with zero-mean output bias and first five weight columns."""

    def __init__(self, d_model: int):
        super().__init__(d_model, d_model)
        basis = self.weight.detach().new_zeros(d_model, d_model - 1)
        for col in range(d_model - 1):
            scale = math.sqrt((col + 1) * (col + 2))
            basis[: col + 1, col] = 1.0 / scale
            basis[col + 1, col] = -(col + 1) / scale
        self.register_buffer("bias_basis", basis, persistent=False)

        full_weight = self.weight.detach()
        leading_weight_coords = basis.transpose(0, 1) @ full_weight[:, :5]
        self.weight = nn.Parameter(
            torch.cat((leading_weight_coords.flatten(), full_weight[:, 5:].flatten()))
        )
        self.bias = nn.Parameter(self.bias.detach()[:-1].clone())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gauge_size = 5 * (self.out_features - 1)
        leading_weight = self.bias_basis @ self.weight[:gauge_size].view(
            self.out_features - 1, 5
        )
        remaining_weight = self.weight[gauge_size:].view(
            self.out_features, self.in_features - 5
        )
        weight = torch.cat((leading_weight, remaining_weight), dim=1)
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
            baseline_weight = module.weight.new_empty(module.weight.numel() + 1)
            nn.init.normal_(baseline_weight, mean=0.0, std=0.02)
            baseline_leading = baseline_weight[
                : 4 * (module.out_features - 1)
            ].view(module.out_features - 1, 4)
            baseline_remaining = baseline_weight[
                4 * (module.out_features - 1) :
            ].view(module.out_features, module.in_features - 4)
            fifth_weight_coords = (
                module.bias_basis.transpose(0, 1)
                @ baseline_remaining[:, :1]
            )
            compact_weight = torch.cat(
                (
                    torch.cat(
                        (baseline_leading, fifth_weight_coords), dim=1
                    ).flatten(),
                    baseline_remaining[:, 1:].flatten(),
                )
            )
            with torch.no_grad():
                module.weight.copy_(compact_weight)
                module.bias.zero_()
        elif isinstance(module, OutputAnchoredLinear):
>>>>>>> REPLACE