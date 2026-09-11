MECHANISM: Orthonormal mean-zero MLP residual quotient

HYPOTHESIS: Representing a second MLP output column in an orthonormal mean-zero basis will produce a 1566-parameter model with at least 99% accuracy, because it removes the same exact feature-uniform residual gauge as the failed coordinate anchor while avoiding its ill-conditioned optimization geometry.

INTENDED_EDIT: Retain the verified `fc2.weight[0,0]` anchor, represent column 4 with seven learned orthonormal mean-zero coordinates, and initialize both constrained columns from a function-equivalent ordinary initialization.

EVIDENCE: The first MLP residual anchor reached 99.46% at 1567 parameters, while adding a second fixed-coordinate anchor at column 4 reached 92.75%; the near-threshold result motivates testing whether quotient-coordinate conditioning, rather than lost functional capacity, caused the failure.

<<<<<<< SEARCH
class FinalBiasAnchoredLinear(nn.Module):
    """MLP output projection with one residual-gauge weight and bias anchor."""

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # Preserve the random-number consumption of nn.Linear construction.
        nn.Linear(in_features, out_features, bias=True)
        self.weight_rest = nn.Parameter(
            torch.empty(out_features * in_features - 1)
        )
        self.bias_rest = nn.Parameter(torch.zeros(out_features - 1))

    def weight_tensor(self) -> torch.Tensor:
        zero = self.weight_rest.new_zeros(1)
        return torch.cat((zero, self.weight_rest)).view(
            self.out_features, self.in_features
        )
=======
class FinalBiasAnchoredLinear(nn.Module):
    """MLP output projection with two residual-gauge columns and a bias anchor."""

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # Preserve the random-number consumption of nn.Linear construction.
        nn.Linear(in_features, out_features, bias=True)
        self.weight_rest = nn.Parameter(
            torch.empty(out_features * in_features - 2)
        )
        self.bias_rest = nn.Parameter(torch.zeros(out_features - 1))

    def gauge_basis(self) -> torch.Tensor:
        unit = self.weight_rest.new_full(
            (self.out_features,), 1.0 / math.sqrt(self.out_features)
        )
        direction = self.weight_rest.new_zeros(self.out_features)
        direction[0] = 1.0
        direction = direction - unit
        eye = torch.eye(
            self.out_features,
            device=self.weight_rest.device,
            dtype=self.weight_rest.dtype,
        )
        reflector = eye - (
            2.0
            * torch.outer(direction, direction)
            / torch.dot(direction, direction)
        )
        return reflector[:, 1:]

    def weight_tensor(self) -> torch.Tensor:
        width = self.out_features - 1
        zero = self.weight_rest.new_zeros(1)
        column_zero = torch.cat((zero, self.weight_rest[:width]))
        column_four = (
            self.gauge_basis()
            @ self.weight_rest[width : 2 * width]
        )
        other = self.weight_rest[2 * width :].view(
            self.out_features, self.in_features - 2
        )
        return torch.cat(
            (
                column_zero.unsqueeze(1),
                other[:, :3],
                column_four.unsqueeze(1),
                other[:, 3:],
            ),
            dim=1,
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
        elif isinstance(module, FinalBiasAnchoredLinear):
            weight = module.weight_rest.new_empty(
                module.out_features, module.in_features
            )
            nn.init.normal_(weight, mean=0.0, std=0.02)
            with torch.no_grad():
                # Removing this coefficient from every output row changes the
                # residual stream only by a feature-uniform, tokenwise shift.
                offset = weight[0, 0].clone()
                weight[:, 0].sub_(offset)
                weight[0, 0] = 0.0
                module.weight_rest.copy_(weight.flatten()[1:])
                nn.init.zeros_(module.bias_rest)
=======
        elif isinstance(module, FinalBiasAnchoredLinear):
            weight = module.weight_rest.new_empty(
                module.out_features, module.in_features
            )
            nn.init.normal_(weight, mean=0.0, std=0.02)
            with torch.no_grad():
                # Preserve the verified coordinate anchor for column zero.
                offset = weight[0, 0].clone()
                weight[:, 0].sub_(offset)
                weight[0, 0] = 0.0

                # Remove column four's uniform output component and encode its
                # remaining degrees of freedom in an orthonormal quotient
                # basis, improving conditioning over another zero-row anchor.
                weight[:, 4].sub_(weight[:, 4].mean())
                basis = module.gauge_basis()
                other = torch.cat((weight[:, 1:4], weight[:, 5:]), dim=1)
                module.weight_rest.copy_(
                    torch.cat(
                        (
                            weight[1:, 0],
                            basis.transpose(0, 1) @ weight[:, 4],
                            other.flatten(),
                        )
                    )
                )
                nn.init.zeros_(module.bias_rest)
>>>>>>> REPLACE