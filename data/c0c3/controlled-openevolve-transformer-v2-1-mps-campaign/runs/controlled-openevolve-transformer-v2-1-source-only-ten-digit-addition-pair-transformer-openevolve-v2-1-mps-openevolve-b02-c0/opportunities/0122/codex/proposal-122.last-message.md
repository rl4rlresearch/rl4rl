MECHANISM: Axis-aligned output-column gauge

HYPOTHESIS: Anchoring the last output coordinate of `fc2` column 6 will reduce the model from 1,575 to 1,574 parameters while retaining at least 99% accuracy, because it preserves the exact downstream-LayerNorm function class while letting AdamW optimize seven original-axis weight differences instead of rotated zero-mean coordinates.

INTENDED_EDIT: Represent `fc2` column 6 by seven differences from its omitted final output weight, reconstruct that weight as zero, and preserve the verified design’s 94-scalar initialization draw.

EVIDENCE: The zero-mean column-6 gauge achieved 47.65% and virtual-coordinate Adam reached 74.43%, indicating optimizer-coordinate sensitivity rather than loss of expressivity; the 99.91%-accurate current model already succeeds with axis-aligned anchoring in `fc2` bias and `fc1`.

<<<<<<< SEARCH
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
=======
class OutputAnchoredLinear(nn.Linear):
    """Projection with four zero-mean columns and column six axis-anchored."""

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
        anchored_weight_coords = (
            full_weight[:-1, 6] - full_weight[-1:, 6]
        )
        final_weight_coords = basis.transpose(0, 1) @ full_weight[:, -1:]
        self.weight = nn.Parameter(
            torch.cat(
                (
                    leading_weight_coords.flatten(),
                    full_weight[:, 3:6].flatten(),
                    anchored_weight_coords,
                    full_weight[:, 7:-1].flatten(),
                    final_weight_coords.flatten(),
                )
            )
        )
        self.bias = nn.Parameter(self.bias.detach()[:-1].clone())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gauge_size = 3 * (self.out_features - 1)
        pre_anchor_end = gauge_size + 3 * self.out_features
        anchor_end = pre_anchor_end + self.out_features - 1
        middle_end = anchor_end + self.out_features * (self.in_features - 8)
        leading_weight = self.weight_basis @ self.weight[:gauge_size].view(
            self.out_features - 1, 3
        )
        pre_anchor_weight = self.weight[gauge_size:pre_anchor_end].view(
            self.out_features, 3
        )
        anchored_weight = F.pad(
            self.weight[pre_anchor_end:anchor_end], (0, 1)
        ).view(self.out_features, 1)
        post_anchor_weight = self.weight[anchor_end:middle_end].view(
            self.out_features, self.in_features - 8
        )
        final_weight = self.weight_basis @ self.weight[middle_end:].view(
            self.out_features - 1, 1
        )
        weight = torch.cat(
            (
                leading_weight,
                pre_anchor_weight,
                anchored_weight,
                post_anchor_weight,
                final_weight,
            ),
            dim=1,
        )
        return F.linear(x, weight, F.pad(self.bias, (0, 1)))
>>>>>>> REPLACE

<<<<<<< SEARCH
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
            with torch.no_grad():
                module.weight.copy_(compact_weight)
                module.bias.zero_()
=======
        if isinstance(module, OutputAnchoredLinear):
            baseline_weight = module.weight.new_empty(module.weight.numel() + 3)
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
            anchored_weight_coords = (
                baseline_remaining[:-1, 4] - baseline_remaining[-1:, 4]
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
                    baseline_remaining[:, 1:4].flatten(),
                    anchored_weight_coords,
                    baseline_remaining[:, 5:-1].flatten(),
                    final_weight_coords.flatten(),
                )
            )
            with torch.no_grad():
                module.weight.copy_(compact_weight)
                module.bias.zero_()
>>>>>>> REPLACE