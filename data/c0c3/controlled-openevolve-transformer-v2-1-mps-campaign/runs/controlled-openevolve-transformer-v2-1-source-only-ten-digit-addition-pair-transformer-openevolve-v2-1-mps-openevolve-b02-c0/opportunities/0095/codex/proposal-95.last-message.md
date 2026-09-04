MECHANISM: Nonadjacent terminal-MLP common-mode gauge fixing

HYPOTHESIS: Constraining the final `fc2` weight column to zero mean while preserving the verified 1,580-parameter model’s initial function and RNG consumption will achieve at least 99% accuracy with 1,579 parameters.

INTENDED_EDIT: Keep the first three constrained `fc2` columns, constrain the nonadjacent final column, and initialize all retained coordinates from the same 94-scalar baseline draw used by the successful design.

EVIDENCE: Three constrained `fc2` columns achieved 99.89%, while constraining the adjacent fourth column fell to 74.27%; testing the final column distinguishes a hidden-unit-specific optimization failure from a general limit on this exact downstream-LayerNorm gauge.

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
        trailing_weight_coords = basis.transpose(0, 1) @ full_weight[:, -1:]
        self.weight = nn.Parameter(
            torch.cat(
                (
                    leading_weight_coords.flatten(),
                    full_weight[:, 3:-1].flatten(),
                    trailing_weight_coords.flatten(),
                )
            )
        )
        self.bias = nn.Parameter(self.bias.detach()[:-1].clone())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gauge_size = 3 * (self.out_features - 1)
        middle_size = self.out_features * (self.in_features - 4)
        leading_weight = self.weight_basis @ self.weight[:gauge_size].view(
            self.out_features - 1, 3
        )
        middle_weight = self.weight[
            gauge_size : gauge_size + middle_size
        ].view(self.out_features, self.in_features - 4)
        trailing_weight = self.weight_basis @ self.weight[
            gauge_size + middle_size :
        ].view(self.out_features - 1, 1)
        weight = torch.cat((leading_weight, middle_weight, trailing_weight), dim=1)
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
            with torch.no_grad():
                module.weight.copy_(compact_weight)
                module.bias.zero_()
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
            trailing_weight_coords = (
                module.weight_basis.transpose(0, 1)
                @ baseline_remaining[:, -1:]
            )
            compact_weight = torch.cat(
                (
                    torch.cat(
                        (baseline_leading, third_weight_coords), dim=1
                    ).flatten(),
                    baseline_remaining[:, 1:-1].flatten(),
                    trailing_weight_coords.flatten(),
                )
            )
            with torch.no_grad():
                module.weight.copy_(compact_weight)
                module.bias.zero_()
>>>>>>> REPLACE