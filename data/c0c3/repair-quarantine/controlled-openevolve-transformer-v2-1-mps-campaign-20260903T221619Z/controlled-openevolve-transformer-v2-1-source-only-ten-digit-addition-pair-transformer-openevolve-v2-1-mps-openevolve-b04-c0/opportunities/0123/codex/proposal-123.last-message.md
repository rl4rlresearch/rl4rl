MECHANISM: Orthonormal mean-zero attention-output residual quotient

HYPOTHESIS: Representing attention projection column 1 in an orthonormal mean-zero basis will produce a 1565-parameter model with at least 99% accuracy, because it removes an exact feature-uniform residual gauge while avoiding the poorly conditioned fixed-coordinate parameterization previously tested on that column.

INTENDED_EDIT: Retain the verified column-0 and column-4 projection anchors, constrain column 1 with seven learned orthonormal mean-zero coordinates, and canonicalize its ordinary initialization into that basis.

EVIDENCE: The fixed `proj.weight[0,1]` anchor collapsed to 15.51%, while replacing the analogous failed second fixed-coordinate MLP anchor with an orthonormal mean-zero quotient raised accuracy from 92.75% to 99.99%; this directly tests whether conditioning likewise explains the projection-column failure.

<<<<<<< SEARCH
class ResidualGaugeLinear(nn.Module):
    """Projection with two weights fixed along feature-uniform residual gauges."""

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # Preserve the random-number consumption of nn.Linear construction.
        nn.Linear(in_features, out_features, bias=True)
        self.weight_rest = nn.Parameter(
            torch.empty(out_features * in_features - 2)
        )
        self.bias = None

    def weight_tensor(self) -> torch.Tensor:
        zero = self.weight_rest.new_zeros(1)
        return torch.cat(
            (zero, self.weight_rest[:3], zero, self.weight_rest[3:])
        ).view(self.out_features, self.in_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight_tensor(), self.bias)
=======
class ResidualGaugeLinear(nn.Module):
    """Projection with three feature-uniform residual-gauge constraints."""

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # Preserve the random-number consumption of nn.Linear construction.
        nn.Linear(in_features, out_features, bias=True)
        self.weight_rest = nn.Parameter(
            torch.empty(out_features * in_features - 3)
        )
        self.bias = None

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
        column_one = (
            self.gauge_basis()
            @ self.weight_rest[width : 2 * width]
        )
        column_four = torch.cat(
            (zero, self.weight_rest[2 * width : 3 * width])
        )
        other = self.weight_rest[3 * width :].view(
            self.out_features, self.in_features - 3
        )
        return torch.cat(
            (
                column_zero.unsqueeze(1),
                column_one.unsqueeze(1),
                other[:, :2],
                column_four.unsqueeze(1),
                other[:, 2:],
            ),
            dim=1,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight_tensor(), self.bias)
>>>>>>> REPLACE

<<<<<<< SEARCH
        elif isinstance(module, ResidualGaugeLinear):
            weight = module.weight_rest.new_empty(
                module.out_features, module.in_features
            )
            nn.init.normal_(weight, mean=0.0, std=0.02)
            with torch.no_grad():
                # Subtracting either input-column coefficient from every
                # output row adds only a feature-uniform residual shift.
                # Subsequent LayerNorms remove both shifts exactly.
                offset = weight[0, 0].clone()
                weight[:, 0].sub_(offset)
                weight[0, 0] = 0.0

                offset = weight[0, 4].clone()
                weight[:, 4].sub_(offset)
                weight[0, 4] = 0.0

                flat = weight.flatten()
                module.weight_rest.copy_(
                    torch.cat((flat[1:4], flat[5:]))
                )
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
=======
        elif isinstance(module, ResidualGaugeLinear):
            weight = module.weight_rest.new_empty(
                module.out_features, module.in_features
            )
            nn.init.normal_(weight, mean=0.0, std=0.02)
            with torch.no_grad():
                # Preserve the two verified coordinate anchors.
                offset = weight[0, 0].clone()
                weight[:, 0].sub_(offset)
                weight[0, 0] = 0.0

                offset = weight[0, 4].clone()
                weight[:, 4].sub_(offset)
                weight[0, 4] = 0.0

                # Encode column one's mean-zero component in an orthonormal
                # quotient basis rather than adding another coordinate anchor.
                weight[:, 1].sub_(weight[:, 1].mean())
                basis = module.gauge_basis()
                other = torch.cat((weight[:, 2:4], weight[:, 5:]), dim=1)
                module.weight_rest.copy_(
                    torch.cat(
                        (
                            weight[1:, 0],
                            basis.transpose(0, 1) @ weight[:, 1],
                            weight[1:, 4],
                            other.flatten(),
                        )
                    )
                )
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
>>>>>>> REPLACE