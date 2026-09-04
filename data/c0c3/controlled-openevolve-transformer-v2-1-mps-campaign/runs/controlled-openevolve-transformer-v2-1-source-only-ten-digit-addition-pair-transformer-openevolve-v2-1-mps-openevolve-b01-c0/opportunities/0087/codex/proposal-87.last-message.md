MECHANISM: Orthonormal fourth attention-output common-mode gauge

HYPOTHESIS: Representing the fourth attention projection column in an orthonormal zero-mean basis will achieve at least 99% accuracy with 1596 parameters because it removes the same downstream LayerNorm-invariant direction without the poorly conditioned coordinate anchor that scored 84.5%.

INTENDED_EDIT: Compress the fourth projection column to seven learned orthonormal coordinates, reconstruct its zero-mean eight-dimensional column, and retain full storage for columns five through eight.

EVIDENCE: Final-coordinate anchoring succeeded for the first three projection columns at 1597 parameters but failed only when extended to the fourth; this motivates testing whether the fourth column needs a better-conditioned gauge chart rather than its redundant common-mode parameter.

<<<<<<< SEARCH
        self.third_column = nn.Parameter(torch.empty(d_model - 1))
        self.rest = nn.Parameter(torch.empty(d_model, d_model - 3))
=======
        self.third_column = nn.Parameter(torch.empty(d_model - 1))
        self.fourth_column = nn.Parameter(torch.empty(d_model - 1))
        self.rest = nn.Parameter(torch.empty(d_model, d_model - 4))

        fourth_basis = torch.zeros(d_model, d_model - 1)
        for column in range(d_model - 1):
            scale = math.sqrt((column + 1) * (column + 2))
            fourth_basis[: column + 1, column] = 1.0 / scale
            fourth_basis[column + 1, column] = -(column + 1) / scale
        self.register_buffer("fourth_basis", fourth_basis, persistent=False)
>>>>>>> REPLACE

<<<<<<< SEARCH
        third_column = weight[:, 2]
        with torch.no_grad():
            self.first_column.copy_(
                first_column[:-1] - first_column[-1]
            )
            self.second_column.copy_(
                second_column[:-1] - second_column[-1]
            )
            self.third_column.copy_(
                third_column[:-1] - third_column[-1]
            )
            self.rest.copy_(weight[:, 3:])
=======
        third_column = weight[:, 2]
        fourth_column = weight[:, 3]
        with torch.no_grad():
            self.first_column.copy_(
                first_column[:-1] - first_column[-1]
            )
            self.second_column.copy_(
                second_column[:-1] - second_column[-1]
            )
            self.third_column.copy_(
                third_column[:-1] - third_column[-1]
            )
            self.fourth_column.copy_(self.fourth_basis.T @ fourth_column)
            self.rest.copy_(weight[:, 4:])
>>>>>>> REPLACE

<<<<<<< SEARCH
        third_column = torch.cat(
            (self.third_column, self.third_column.new_zeros(1))
        )
        return torch.cat(
            (
                first_column.unsqueeze(1),
                second_column.unsqueeze(1),
                third_column.unsqueeze(1),
                self.rest,
            ),
            dim=1,
        )
=======
        third_column = torch.cat(
            (self.third_column, self.third_column.new_zeros(1))
        )
        fourth_column = self.fourth_basis @ self.fourth_column
        return torch.cat(
            (
                first_column.unsqueeze(1),
                second_column.unsqueeze(1),
                third_column.unsqueeze(1),
                fourth_column.unsqueeze(1),
                self.rest,
            ),
            dim=1,
        )
>>>>>>> REPLACE