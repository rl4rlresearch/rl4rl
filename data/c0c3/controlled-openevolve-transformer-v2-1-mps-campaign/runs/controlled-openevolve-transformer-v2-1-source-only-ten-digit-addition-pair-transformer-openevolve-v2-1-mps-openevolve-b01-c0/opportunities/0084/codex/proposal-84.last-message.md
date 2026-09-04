MECHANISM: Second attention-output common-mode gauge anchor

HYPOTHESIS: Anchoring the final output coordinate of the second attention projection column will reduce the model to 1598 parameters while retaining at least 99% accuracy, because it removes the same downstream LayerNorm-invariant common-mode direction as the successfully anchored first column.

INTENDED_EDIT: Store seven coordinates for each of the first two attention output-projection columns, reconstruct both eighth coordinates as zero, and preserve initialization modulo feature-common residual shifts.

EVIDENCE: Anchoring the first attention projection column achieved 99.83% accuracy at 1599 parameters; extending that exact successful parameterization to the adjacent column is the most direct one-parameter reduction.

<<<<<<< SEARCH
class AnchoredOutputLinear(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        # Consume the same constructor RNG as the original projection layer.
        _ = nn.Linear(d_model, d_model)
        self.d_model = d_model
        self.first_column = nn.Parameter(torch.empty(d_model - 1))
        self.rest = nn.Parameter(torch.empty(d_model, d_model - 1))

    def _set_from_full(self, weight: torch.Tensor) -> None:
        first_column = weight[:, 0]
        with torch.no_grad():
            self.first_column.copy_(
                first_column[:-1] - first_column[-1]
            )
            self.rest.copy_(weight[:, 1:])

    def reconstructed_weight(self) -> torch.Tensor:
        first_column = torch.cat(
            (self.first_column, self.first_column.new_zeros(1))
        )
        return torch.cat((first_column.unsqueeze(1), self.rest), dim=1)
=======
class AnchoredOutputLinear(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        # Consume the same constructor RNG as the original projection layer.
        _ = nn.Linear(d_model, d_model)
        self.d_model = d_model
        self.first_column = nn.Parameter(torch.empty(d_model - 1))
        self.second_column = nn.Parameter(torch.empty(d_model - 1))
        self.rest = nn.Parameter(torch.empty(d_model, d_model - 2))

    def _set_from_full(self, weight: torch.Tensor) -> None:
        first_column = weight[:, 0]
        second_column = weight[:, 1]
        with torch.no_grad():
            self.first_column.copy_(
                first_column[:-1] - first_column[-1]
            )
            self.second_column.copy_(
                second_column[:-1] - second_column[-1]
            )
            self.rest.copy_(weight[:, 2:])

    def reconstructed_weight(self) -> torch.Tensor:
        first_column = torch.cat(
            (self.first_column, self.first_column.new_zeros(1))
        )
        second_column = torch.cat(
            (self.second_column, self.second_column.new_zeros(1))
        )
        return torch.cat(
            (
                first_column.unsqueeze(1),
                second_column.unsqueeze(1),
                self.rest,
            ),
            dim=1,
        )
>>>>>>> REPLACE