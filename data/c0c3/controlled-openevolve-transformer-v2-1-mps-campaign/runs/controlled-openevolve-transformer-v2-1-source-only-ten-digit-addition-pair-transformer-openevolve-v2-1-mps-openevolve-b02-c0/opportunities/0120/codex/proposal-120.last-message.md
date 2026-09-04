MECHANISM: Central MLP-output common-mode gauge

HYPOTHESIS: Constraining `fc2` column 6 (the seventh column) to zero mean will reduce the model from 1,575 to 1,574 parameters while retaining at least 99% accuracy, because its removed all-ones component is invisible to the downstream final LayerNorm and the central coordinate avoids the optimization-sensitive adjacent fourth and penultimate columns.

INTENDED_EDIT: Preserve the verified gauges on the first three and final `fc2` columns, additionally represent central column 6 in the zero-mean basis, and retain the same 94-scalar initialization draw.

EVIDENCE: The nonadjacent final-column `fc2` gauge achieved 99.91% at 1,575 parameters, while adding gauges at the adjacent fourth or penultimate columns failed; testing an isolated central column is the most direct next probe of coordinate-specific optimization sensitivity within the established exact gauge.

<<<<<<< SEARCH
    """Linear projection with anchored bias and four zero-mean weight columns."""
=======
    """Linear projection with anchored bias and five zero-mean weight columns."""
>>>>>>> REPLACE

<<<<<<< SEARCH
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
=======
        full_weight = self.weight.detach()
        leading_weight_coords = basis.transpose(0, 1) @ full_weight[:, :3]
        central_weight_coords = basis.transpose(0, 1) @ full_weight[:, 6:7]
        final_weight_coords = basis.transpose(0, 1) @ full_weight[:, -1:]
        self.weight = nn.Parameter(
            torch.cat(
                (
                    leading_weight_coords.flatten(),
                    full_weight[:, 3:6].flatten(),
                    central_weight_coords.flatten(),
                    full_weight[:, 7:-1].flatten(),
                    final_weight_coords.flatten(),
                )
            )
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
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
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gauge_size = 3 * (self.out_features - 1)
        pre_central_end = gauge_size + 3 * self.out_features
        central_end = pre_central_end + self.out_features - 1
        post_central_end = central_end + self.out_features * (
            self.in_features - 8
        )
        leading_weight = self.weight_basis @ self.weight[:gauge_size].view(
            self.out_features - 1, 3
        )
        pre_central_weight = self.weight[gauge_size:pre_central_end].view(
            self.out_features, 3
        )
        central_weight = self.weight_basis @ self.weight[
            pre_central_end:central_end
        ].view(self.out_features - 1, 1)
        post_central_weight = self.weight[
            central_end:post_central_end
        ].view(self.out_features, self.in_features - 8)
        final_weight = self.weight_basis @ self.weight[
            post_central_end:
        ].view(self.out_features - 1, 1)
        weight = torch.cat(
            (
                leading_weight,
                pre_central_weight,
                central_weight,
                post_central_weight,
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
            central_weight_coords = (
                module.weight_basis.transpose(0, 1)
                @ baseline_remaining[:, 4:5]
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
                    central_weight_coords.flatten(),
                    baseline_remaining[:, 5:-1].flatten(),
                    final_weight_coords.flatten(),
                )
            )
>>>>>>> REPLACE