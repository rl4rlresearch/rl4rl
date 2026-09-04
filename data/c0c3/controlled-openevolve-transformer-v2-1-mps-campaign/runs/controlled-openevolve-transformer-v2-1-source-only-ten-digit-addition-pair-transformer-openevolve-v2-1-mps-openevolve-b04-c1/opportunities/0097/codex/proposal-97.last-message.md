MECHANISM: Biased-head stabilizer shear gauge

HYPOTHESIS: Quotienting one query-weight coordinate in the biased attention head will reduce the model from 869 to 868 learned parameters while retaining at least 99% accuracy, because a Q/K shear that leaves its one-axis query bias unchanged preserves every attention logit.

INTENDED_EDIT: Triangularize one coordinate of the biased head’s first query row using its unbiased second query row, apply the compensating inverse-transpose transformation to its keys, and omit the resulting zero coordinate from training.

EVIDENCE: The unbiased-head shear gauge retained 99.93% accuracy at 869 parameters; the biased head admits the same exact shear within the subgroup that preserves its existing one-coordinate query bias.

<<<<<<< SEARCH
class NarrowQKVLinear(nn.Module):
    """Input-shift gauges plus a joint Q/K rotation gauge."""

    def __init__(
        self, d_model: int, n_head: int, qk_dim: int
    ):
        super().__init__()
        self.d_model = d_model
        self.n_head = n_head
        self.head_dim = d_model // n_head
        self.qk_dim = qk_dim
        self.out_features = 2 * n_head * qk_dim + d_model
        # The second head has no query bias. Its full Q/K change-of-basis
        # symmetry permits a rotation zero followed by an independent shear
        # zero in the two query rows.
        self.shear_gauge_row = qk_dim
        self.rotation_gauge_row = qk_dim + 1
        self.coordinates = nn.ParameterList(
            [
                nn.Parameter(
                    torch.empty(
                        d_model - 2
                        if row in (
                            self.shear_gauge_row,
                            self.rotation_gauge_row,
                        )
                        else d_model - 1
                    )
                )
                for row in range(self.out_features)
            ]
        )
        self.register_buffer(
            "shear_pivot", torch.tensor(2, dtype=torch.long)
        )
=======
class NarrowQKVLinear(nn.Module):
    """Input-shift gauges plus exact within-head Q/K gauges."""

    def __init__(
        self, d_model: int, n_head: int, qk_dim: int
    ):
        super().__init__()
        self.d_model = d_model
        self.n_head = n_head
        self.head_dim = d_model // n_head
        self.qk_dim = qk_dim
        self.out_features = 2 * n_head * qk_dim + d_model
        # The first head's one-axis query bias is preserved by an upper
        # shear. The unbiased second head additionally permits the existing
        # rotation and independent shear gauges.
        self.biased_shear_gauge_row = 0
        self.shear_gauge_row = qk_dim
        self.rotation_gauge_row = qk_dim + 1
        self.coordinates = nn.ParameterList(
            [
                nn.Parameter(
                    torch.empty(
                        d_model - 2
                        if row in (
                            self.biased_shear_gauge_row,
                            self.shear_gauge_row,
                            self.rotation_gauge_row,
                        )
                        else d_model - 1
                    )
                )
                for row in range(self.out_features)
            ]
        )
        self.register_buffer(
            "biased_shear_pivot", torch.tensor(1, dtype=torch.long)
        )
        self.register_buffer(
            "shear_pivot", torch.tensor(2, dtype=torch.long)
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
            q0 = self.qk_dim
            q1 = q0 + 1
            qk_width = self.n_head * self.qk_dim
            k0 = qk_width + q0
            k1 = k0 + 1

            # Rotate the unbiased head so the first stored coordinate of its
=======
            qk_width = self.n_head * self.qk_dim

            # Shear the biased head without changing its query bias: only its
            # first query channel has a bias, while the second has none.
            biased_q0 = 0
            biased_q1 = 1
            biased_k0 = qk_width
            biased_k1 = qk_width + 1
            biased_pivot = int(
                torch.argmax(transformed[biased_q1, 1:].abs()).item()
            ) + 1
            self.biased_shear_pivot.fill_(biased_pivot)
            biased_shear = (
                transformed[biased_q0, biased_pivot]
                / transformed[biased_q1, biased_pivot]
            )
            transformed[biased_q0, 1:] = (
                transformed[biased_q0, 1:]
                - biased_shear * transformed[biased_q1, 1:]
            )
            transformed[biased_k1, 1:] = (
                transformed[biased_k1, 1:]
                + biased_shear * transformed[biased_k0, 1:]
            )

            q0 = self.qk_dim
            q1 = q0 + 1
            k0 = qk_width + q0
            k1 = k0 + 1

            # Rotate the unbiased head so the first stored coordinate of its
>>>>>>> REPLACE

<<<<<<< SEARCH
                if row_index == self.shear_gauge_row:
                    coordinates.copy_(
                        torch.cat((row[1:pivot], row[pivot + 1 :]))
                    )
                elif row_index == self.rotation_gauge_row:
=======
                if row_index == self.biased_shear_gauge_row:
                    coordinates.copy_(
                        torch.cat(
                            (
                                row[1:biased_pivot],
                                row[biased_pivot + 1 :],
                            )
                        )
                    )
                elif row_index == self.shear_gauge_row:
                    coordinates.copy_(
                        torch.cat((row[1:pivot], row[pivot + 1 :]))
                    )
                elif row_index == self.rotation_gauge_row:
>>>>>>> REPLACE

<<<<<<< SEARCH
    def ambient_coordinate_indices(
        self, row_index: int
    ) -> Tuple[int, ...]:
        if row_index == self.shear_gauge_row:
            pivot = int(self.shear_pivot.item())
=======
    def ambient_coordinate_indices(
        self, row_index: int
    ) -> Tuple[int, ...]:
        if row_index == self.biased_shear_gauge_row:
            pivot = int(self.biased_shear_pivot.item())
            return tuple(
                index
                for index in range(1, self.d_model)
                if index != pivot
            )
        if row_index == self.shear_gauge_row:
            pivot = int(self.shear_pivot.item())
>>>>>>> REPLACE

<<<<<<< SEARCH
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        transformed_rows = []
        pivot = int(self.shear_pivot.item())
        for row_index, coordinates in enumerate(self.coordinates):
            if row_index == self.shear_gauge_row:
                split = pivot - 1
                row = torch.cat(
                    (
                        coordinates.new_zeros(1),
                        coordinates[:split],
                        coordinates.new_zeros(1),
                        coordinates[split:],
                    )
                )
            elif row_index == self.rotation_gauge_row:
=======
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        transformed_rows = []
        biased_pivot = int(self.biased_shear_pivot.item())
        pivot = int(self.shear_pivot.item())
        for row_index, coordinates in enumerate(self.coordinates):
            if row_index == self.biased_shear_gauge_row:
                split = biased_pivot - 1
                row = torch.cat(
                    (
                        coordinates.new_zeros(1),
                        coordinates[:split],
                        coordinates.new_zeros(1),
                        coordinates[split:],
                    )
                )
            elif row_index == self.shear_gauge_row:
                split = pivot - 1
                row = torch.cat(
                    (
                        coordinates.new_zeros(1),
                        coordinates[:split],
                        coordinates.new_zeros(1),
                        coordinates[split:],
                    )
                )
            elif row_index == self.rotation_gauge_row:
>>>>>>> REPLACE