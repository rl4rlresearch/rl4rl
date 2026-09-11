MECHANISM: Second-head key-row anchoring

HYPOTHESIS: Anchoring the final input coordinate of the fifth key-projection row will retain at least 99% accuracy with 1599 parameters, because it applies the successful anchored gauge to the first key coordinate of the previously unconstrained second attention head.

INTENDED_EDIT: Compress key rows one, two, and five to seven learned coordinates each, reconstructing their eighth coordinates as zero while preserving initialization-equivalent row transformations.

EVIDENCE: Anchoring the first two key rows achieved 99.93% at 1600 parameters, while anchoring the third or fourth rows produced about 74% accuracy; because those four rows comprise the first attention head, testing the fifth row isolates whether the failure is specific to later coordinates of that head.

<<<<<<< SEARCH
        self.before_key = nn.Parameter(torch.empty(d_model, d_model))
        self.key_first_row = nn.Parameter(torch.empty(d_model - 1))
        self.key_second_row = nn.Parameter(torch.empty(d_model - 1))
        self.after_key = nn.Parameter(torch.empty(2 * d_model - 2, d_model))
=======
        self.before_key = nn.Parameter(torch.empty(d_model, d_model))
        self.key_first_row = nn.Parameter(torch.empty(d_model - 1))
        self.key_second_row = nn.Parameter(torch.empty(d_model - 1))
        self.after_second_before_fifth = nn.Parameter(torch.empty(2, d_model))
        self.key_fifth_row = nn.Parameter(torch.empty(d_model - 1))
        self.after_fifth = nn.Parameter(torch.empty(2 * d_model - 5, d_model))
>>>>>>> REPLACE

<<<<<<< SEARCH
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
=======
        first_key_row = weight[self.d_model]
        second_key_row = weight[self.d_model + 1]
        fifth_key_row = weight[self.d_model + 4]
        with torch.no_grad():
            self.before_key.copy_(weight[: self.d_model])
            self.key_first_row.copy_(
                first_key_row[:-1] - first_key_row[-1]
            )
            self.key_second_row.copy_(
                second_key_row[:-1] - second_key_row[-1]
            )
            self.after_second_before_fifth.copy_(
                weight[self.d_model + 2 : self.d_model + 4]
            )
            self.key_fifth_row.copy_(
                fifth_key_row[:-1] - fifth_key_row[-1]
            )
            self.after_fifth.copy_(weight[self.d_model + 5 :])
>>>>>>> REPLACE

<<<<<<< SEARCH
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
        second_key_row = torch.cat(
            (self.key_second_row, self.key_second_row.new_zeros(1))
        )
        fifth_key_row = torch.cat(
            (self.key_fifth_row, self.key_fifth_row.new_zeros(1))
        )
        return torch.cat(
            (
                self.before_key,
                first_key_row.unsqueeze(0),
                second_key_row.unsqueeze(0),
                self.after_second_before_fifth,
                fifth_key_row.unsqueeze(0),
                self.after_fifth,
            ),
            dim=0,
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
        # Preserve the original constructor RNG while anchoring one redundant
        # coordinate in each of the first two key rows.
=======
        # Preserve the original constructor RNG while anchoring one redundant
        # coordinate in the first, second, and fifth key rows.
>>>>>>> REPLACE