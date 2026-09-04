MECHANISM: Nonadjacent MLP-output common-mode gauge

HYPOTHESIS: Constraining the final `fc2` output-projection column to zero mean will reduce the model to 1,575 parameters while retaining at least 99% accuracy, because this exact downstream-LayerNorm gauge avoids the activation coordinate implicated by the failed fourth-column extension.

INTENDED_EDIT: Preserve the three verified leading `fc2` gauges, additionally represent only the final column in the zero-mean basis, and initialize it by projecting the same 94-scalar baseline draw as the 1,576-parameter design.

EVIDENCE: Three gauged `fc2` columns contributed to the 99.92%-accurate 1,576-parameter model, while extending the gauge specifically to the adjacent fourth column achieved only 73.92%; testing the nonadjacent final column distinguishes coordinate-specific optimization sensitivity from failure of the established gauge mechanism.

<<<<<<< SEARCH
class OutputAnchoredLinear(nn.Linear):
    """Linear projection with anchored bias and three zero-mean weight columns."""

    def __init__(self, in_features: int, out_features: int):
        super().__init__(in_features, out_features)
        basis = self.weight.detach().new_zeros(out_features, out_features - 1)
        for col in range(out_features - 1):
            scale = math.sqrt((col + 1) * (col + 2))
            basis[: col + 1, col] = 1.0 / scale
            basis[col + 1, col] = -(col + 1) / scale
        self.register_buffer("weight_basis", basis, persistent=False)

        full_weight = self.weight.detach()
        leading_weight_coords = basis.transpose(0, 1) @ full_weight[:, :3]
        self.weight = nn.Parameter(
            torch.cat((leading_weight_coords.flatten(), full_weight[:, 3:].flatten()))
        )
        self.bias = nn.Parameter(self.bias.detach()[:-1].clone())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gauge_size = 3 * (self.out_features - 1)
        leading_weight = self.weight_basis @ self.weight[:gauge_size].view(
            self.out_features - 1, 3
        )
        remaining_weight = self.weight[gauge_size:].view(
            self.out_features, self.in_features - 3
        )
        weight = torch.cat((leading_weight, remaining_weight), dim=1)
        return F.linear(x, weight, F.pad(self.bias, (0, 1)))
=======
class OutputAnchoredLinear(nn.Linear):
    """Linear projection with anchored bias and four zero-mean weight columns."""

    def __init__(self, in_features: int, out_features: int):
        super().__init__(in_features, out_features)
        basis = self.weight.detach().new_zeros(out_features, out_features - 1)
        for col in range(out_features - 1):
            scale = math.sqrt((col + 1) * (col + 2))
            basis[: col + 1, col] = 1.0 / scale
            basis[col + 1, col] = -(col + 1) / scale
        self.register_buffer("weight_basis", basis, persistent=False)

        full_weight = self.weight.detach()
        leading_weight_coords = basis.transpose(0, 1) @ full_weight[:, :3]
        final_weight_coords = basis.transpose(0, 1) @ full_weight[:, -1:]
        self.weight = nn.Parameter(
            torch.cat(
                (
                    leading_weight_coords.flatten(),
                    full_weight[:, 3:-1].flatten(),
                    final_weight_coords.flatten(),
                )
            )
        )
        self.bias = nn.Parameter(self.bias.detach()[:-1].clone())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gauge_size = 3 * (self.out_features - 1)
        middle_end = gauge_size + self.out_features * (self.in_features - 4)
        leading_weight = self.weight_basis @ self.weight[:gauge_size].view(
            self.out_features - 1, 3
        )
        middle_weight = self.weight[gauge_size:middle_end].view(
            self.out_features, self.in_features - 4
        )
        final_weight = self.weight_basis @ self.weight[middle_end:].view(
            self.out_features - 1, 1
        )
        weight = torch.cat((leading_weight, middle_weight, final_weight), dim=1)
        return F.linear(x, weight, F.pad(self.bias, (0, 1)))
>>>>>>> REPLACE

<<<<<<< SEARCH
        if isinstance(module, OutputAnchoredLinear):
            baseline_weight = module.weight.new_empty(module.weight.numel() + 1)
            nn.init.normal_(baseline_weight, mean=0.0, std=0.02)
            baseline_leading = baseline_weight[
                : 2 * (module.out_features - 1)
            ].view(module.out_features - 1, 2)
            baseline_remaining = baseline_weight[
                2 * (module.out_features - 1) :
            ].view(module.out_features, module.in_features - 2)
            third_weight_coords = (
                module.weight_basis.transpose(0, 1)
                @ baseline_remaining[:, :1]
            )
            compact_weight = torch.cat(
                (
                    torch.cat(
                        (baseline_leading, third_weight_coords), dim=1
                    ).flatten(),
                    baseline_remaining[:, 1:].flatten(),
                )
            )
=======
        if isinstance(module, OutputAnchoredLinear):
            baseline_weight = module.weight.new_empty(module.weight.numel() + 2)
            nn.init.normal_(baseline_weight, mean=0.0, std=0.02)
            baseline_leading = baseline_weight[
                : 2 * (module.out_features - 1)
            ].view(module.out_features - 1, 2)
            baseline_remaining = baseline_weight[
                2 * (module.out_features - 1) :
            ].view(module.out_features, module.in_features - 2)
            third_weight_coords = (
                module.weight_basis.transpose(0, 1)
                @ baseline_remaining[:, :1]
            )
            final_weight_coords = (
                module.weight_basis.transpose(0, 1)
                @ baseline_remaining[:, -1:]
            )
            compact_weight = torch.cat(
                (
                    torch.cat(
                        (baseline_leading, third_weight_coords), dim=1
                    ).flatten(),
                    baseline_remaining[:, 1:-1].flatten(),
                    final_weight_coords.flatten(),
                )
            )
>>>>>>> REPLACE