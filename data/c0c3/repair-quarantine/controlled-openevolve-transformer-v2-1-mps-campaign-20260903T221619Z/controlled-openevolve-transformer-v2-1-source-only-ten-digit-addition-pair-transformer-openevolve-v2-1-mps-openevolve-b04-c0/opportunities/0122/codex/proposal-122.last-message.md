MECHANISM: Third orthonormal MLP residual quotient

HYPOTHESIS: Representing MLP output column 8 in the same orthonormal mean-zero basis as column 4 will reduce the model to 1565 parameters while retaining at least 99% accuracy, because it removes another exact feature-uniform residual gauge without introducing a poorly conditioned coordinate anchor.

INTENDED_EDIT: Retain the existing column-0 and column-4 constraints, encode column 8 with seven learned mean-zero coordinates, and canonicalize its ordinary initialization.

EVIDENCE: Replacing the failed second fixed-coordinate MLP anchor with an orthonormal mean-zero representation improved accuracy from 92.75% to 99.99% at 1566 parameters; applying that successful parameterization to the nonadjacent column 8 is the closest supported reduction.

<<<<<<< SEARCH
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
=======
class FinalBiasAnchoredLinear(nn.Module):
    """MLP output projection with three residual-gauge columns and a bias anchor."""

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # Preserve the random-number consumption of nn.Linear construction.
        nn.Linear(in_features, out_features, bias=True)
        self.weight_rest = nn.Parameter(
            torch.empty(out_features * in_features - 3)
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
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
=======
        column_four = (
            self.gauge_basis()
            @ self.weight_rest[width : 2 * width]
        )
        column_eight = (
            self.gauge_basis()
            @ self.weight_rest[2 * width : 3 * width]
        )
        other = self.weight_rest[3 * width :].view(
            self.out_features, self.in_features - 3
        )
        return torch.cat(
            (
                column_zero.unsqueeze(1),
                other[:, :3],
                column_four.unsqueeze(1),
                other[:, 3:6],
                column_eight.unsqueeze(1),
                other[:, 6:],
            ),
            dim=1,
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
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
=======
                # Remove columns four and eight's uniform output components
                # and encode their remaining degrees of freedom in the same
                # orthonormal quotient basis.
                weight[:, 4].sub_(weight[:, 4].mean())
                weight[:, 8].sub_(weight[:, 8].mean())
                basis = module.gauge_basis()
                other = torch.cat(
                    (weight[:, 1:4], weight[:, 5:8], weight[:, 9:]), dim=1
                )
                module.weight_rest.copy_(
                    torch.cat(
                        (
                            weight[1:, 0],
                            basis.transpose(0, 1) @ weight[:, 4],
                            basis.transpose(0, 1) @ weight[:, 8],
                            other.flatten(),
                        )
                    )
                )
>>>>>>> REPLACE