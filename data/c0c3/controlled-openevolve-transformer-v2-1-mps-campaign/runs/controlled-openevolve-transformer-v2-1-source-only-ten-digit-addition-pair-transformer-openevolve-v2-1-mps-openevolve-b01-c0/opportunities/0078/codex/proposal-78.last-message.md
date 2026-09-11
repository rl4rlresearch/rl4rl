MECHANISM: Third-row anchored key-projection LayerNorm gauge

HYPOTHESIS: Anchoring the final input coordinate of a third key-projection row will reduce the model to 1599 parameters while maintaining at least 99% accuracy, because the identical anchored parameterization succeeded on the first two key rows.

INTENDED_EDIT: Store seven coordinates for each of the first three key rows, fix their eighth coordinates to zero, and transform all three rows from the original initialization.

EVIDENCE: Anchoring one key row achieved 100% accuracy at 1601 parameters, and extending it to a second row achieved 99.93% at 1600; applying the same successful reduction to the adjacent third row is the most direct one-parameter test.

<<<<<<< SEARCH
class AnchoredKeyLinear(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        # Consume the same constructor RNG as the original QKV linear layer.
        _ = nn.Linear(d_model, 3 * d_model)
        self.d_model = d_model
        self.before_key = nn.Parameter(torch.empty(d_model, d_model))
        self.key_first_row = nn.Parameter(torch.empty(d_model - 1))
        self.key_second_row = nn.Parameter(torch.empty(d_model - 1))
        self.after_key = nn.Parameter(torch.empty(2 * d_model - 2, d_model))

    def _set_from_full(self, weight: torch.Tensor) -> None:
        first_key_row = weight[self.d_model]
        second_key_row = weight[self.d_model + 1]
        with torch.no_grad():
            self.before_key.copy_(weight[: self.d_model])
            self.key_first_row.copy_(
                first_key_row[:-1] - first_key_row[-1]
            )
            self.key_second_row.copy_(
                second_key_row[:-1] - second_key_row[-1]
            )
            self.after_key.copy_(weight[self.d_model + 2 :])

    def reconstructed_weight(self) -> torch.Tensor:
        first_key_row = torch.cat(
            (self.key_first_row, self.key_first_row.new_zeros(1))
        )
        second_key_row = torch.cat(
            (self.key_second_row, self.key_second_row.new_zeros(1))
        )
        return torch.cat(
            (
                self.before_key,
                first_key_row.unsqueeze(0),
                second_key_row.unsqueeze(0),
                self.after_key,
            ),
            dim=0,
        )
=======
class AnchoredKeyLinear(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        # Consume the same constructor RNG as the original QKV linear layer.
        _ = nn.Linear(d_model, 3 * d_model)
        self.d_model = d_model
        self.before_key = nn.Parameter(torch.empty(d_model, d_model))
        self.key_first_row = nn.Parameter(torch.empty(d_model - 1))
        self.key_second_row = nn.Parameter(torch.empty(d_model - 1))
        self.key_third_row = nn.Parameter(torch.empty(d_model - 1))
        self.after_key = nn.Parameter(torch.empty(2 * d_model - 3, d_model))

    def _set_from_full(self, weight: torch.Tensor) -> None:
        first_key_row = weight[self.d_model]
        second_key_row = weight[self.d_model + 1]
        third_key_row = weight[self.d_model + 2]
        with torch.no_grad():
            self.before_key.copy_(weight[: self.d_model])
            self.key_first_row.copy_(
                first_key_row[:-1] - first_key_row[-1]
            )
            self.key_second_row.copy_(
                second_key_row[:-1] - second_key_row[-1]
            )
            self.key_third_row.copy_(
                third_key_row[:-1] - third_key_row[-1]
            )
            self.after_key.copy_(weight[self.d_model + 3 :])

    def reconstructed_weight(self) -> torch.Tensor:
        first_key_row = torch.cat(
            (self.key_first_row, self.key_first_row.new_zeros(1))
        )
        second_key_row = torch.cat(
            (self.key_second_row, self.key_second_row.new_zeros(1))
        )
        third_key_row = torch.cat(
            (self.key_third_row, self.key_third_row.new_zeros(1))
        )
        return torch.cat(
            (
                self.before_key,
                first_key_row.unsqueeze(0),
                second_key_row.unsqueeze(0),
                third_key_row.unsqueeze(0),
                self.after_key,
            ),
            dim=0,
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
        # Preserve the original constructor RNG while anchoring one redundant
        # coordinate in each of the first two key rows.
=======
        # Preserve the original constructor RNG while anchoring one redundant
        # coordinate in each of the first three key rows.
>>>>>>> REPLACE