MECHANISM: Coordinate-selective fourth key-row anchoring

HYPOTHESIS: Anchoring the final input coordinate of the fourth key-projection row will achieve at least 99% accuracy with 1599 parameters, because the first two rows tolerate this reduction while the third row’s failure under two anchoring charts indicates row-specific importance.

INTENDED_EDIT: Keep the third key row fully learned, compress the fourth key row to seven learned coordinates with its eighth fixed at zero, and preserve the initialization-equivalent key transformation.

EVIDENCE: Final-coordinate anchors on the first two key rows achieved 99.93% at 1600 parameters, whereas anchoring either the first or final coordinate of the third row scored about 74%; this motivates testing the adjacent fourth row without constraining the sensitive third row.

<<<<<<< SEARCH
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
        self.before_key = nn.Parameter(torch.empty(d_model, d_model))
        self.key_first_row = nn.Parameter(torch.empty(d_model - 1))
        self.key_second_row = nn.Parameter(torch.empty(d_model - 1))
        self.key_third_row = nn.Parameter(torch.empty(d_model))
        self.key_fourth_row = nn.Parameter(torch.empty(d_model - 1))
        self.after_key = nn.Parameter(torch.empty(2 * d_model - 4, d_model))

    def _set_from_full(self, weight: torch.Tensor) -> None:
        first_key_row = weight[self.d_model]
        second_key_row = weight[self.d_model + 1]
        third_key_row = weight[self.d_model + 2]
        fourth_key_row = weight[self.d_model + 3]
        with torch.no_grad():
            self.before_key.copy_(weight[: self.d_model])
            self.key_first_row.copy_(
                first_key_row[:-1] - first_key_row[-1]
            )
            self.key_second_row.copy_(
                second_key_row[:-1] - second_key_row[-1]
            )
            self.key_third_row.copy_(third_key_row)
            self.key_fourth_row.copy_(
                fourth_key_row[:-1] - fourth_key_row[-1]
            )
            self.after_key.copy_(weight[self.d_model + 4 :])

    def reconstructed_weight(self) -> torch.Tensor:
        first_key_row = torch.cat(
            (self.key_first_row, self.key_first_row.new_zeros(1))
        )
        second_key_row = torch.cat(
            (self.key_second_row, self.key_second_row.new_zeros(1))
        )
        fourth_key_row = torch.cat(
            (self.key_fourth_row, self.key_fourth_row.new_zeros(1))
        )
        return torch.cat(
            (
                self.before_key,
                first_key_row.unsqueeze(0),
                second_key_row.unsqueeze(0),
                self.key_third_row.unsqueeze(0),
                fourth_key_row.unsqueeze(0),
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
        # coordinate in the first, second, and fourth key rows.
>>>>>>> REPLACE