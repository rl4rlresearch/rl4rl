MECHANISM: Unbiased-head Q/K rotational gauge fixing

HYPOTHESIS: Canonicalizing one query-weight coordinate in the unbiased second attention head will reduce the model from 871 to 870 parameters while retaining at least 99% accuracy, because a joint rotation of that head’s query and key channels preserves every attention logit.

INTENDED_EDIT: Remove one coordinate from the second head’s second query row, jointly rotate its initialized Q/K row pairs into the resulting gauge, and extend ambient AdamW to support the additional fixed coordinate.

EVIDENCE: The previous query-bias rotational gauge retained 99.89% accuracy at 871 parameters, while the narrow two-dimensional Q/K design retained 99.92%; the unbiased head has the same exact joint Q/K rotational symmetry without a bias vector restricting it.

<<<<<<< SEARCH
class NarrowQKVLinear(nn.Module):
    """Gauge-fixed Q/K bottleneck with an unchanged full-width value path."""

    def __init__(
        self, d_model: int, n_head: int, qk_dim: int
    ):
        super().__init__()
        self.d_model = d_model
        self.n_head = n_head
        self.head_dim = d_model // n_head
        self.qk_dim = qk_dim
        self.out_features = 2 * n_head * qk_dim + d_model
        self.coordinates = nn.ParameterList(
            [
                nn.Parameter(torch.empty(d_model - 1))
                for _ in range(self.out_features)
            ]
        )
=======
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
        # The second head has no query bias. Jointly rotating its first two
        # Q/K channels therefore permits this Q-row coordinate to be fixed.
        self.rotation_gauge_row = qk_dim + 1
        self.coordinates = nn.ParameterList(
            [
                nn.Parameter(
                    torch.empty(
                        d_model - 2
                        if row == self.rotation_gauge_row
                        else d_model - 1
                    )
                )
                for row in range(self.out_features)
            ]
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
    def _set_weight(self, conceptual_weight: torch.Tensor) -> None:
        with torch.no_grad():
            transformed = self._householder(
                self._select_weight(conceptual_weight)
            )
            for coordinates, row in zip(self.coordinates, transformed):
                coordinates.copy_(row[1:])
=======
    def _set_weight(self, conceptual_weight: torch.Tensor) -> None:
        with torch.no_grad():
            transformed = self._householder(
                self._select_weight(conceptual_weight)
            )

            q0 = self.qk_dim
            q1 = q0 + 1
            qk_width = self.n_head * self.qk_dim
            k0 = qk_width + q0
            k1 = k0 + 1

            # Rotate the unbiased head so the first stored coordinate of its
            # second Q row is zero, applying the identical rotation to K.
            anchor = transformed[q0 : q1 + 1, 1]
            angle = torch.atan2(anchor[1], anchor[0])
            cosine = torch.cos(angle)
            sine = torch.sin(angle)
            rotation = torch.stack(
                (
                    torch.stack((cosine, sine)),
                    torch.stack((-sine, cosine)),
                )
            )
            transformed[q0 : q1 + 1, 1:] = (
                rotation @ transformed[q0 : q1 + 1, 1:]
            )
            transformed[k0 : k1 + 1, 1:] = (
                rotation @ transformed[k0 : k1 + 1, 1:]
            )

            for row_index, (coordinates, row) in enumerate(
                zip(self.coordinates, transformed)
            ):
                if row_index == self.rotation_gauge_row:
                    coordinates.copy_(row[2:])
                else:
                    coordinates.copy_(row[1:])
>>>>>>> REPLACE

<<<<<<< SEARCH
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        coordinates = torch.stack(list(self.coordinates), dim=0)
        weight = self._householder(F.pad(coordinates, (1, 0)))
        return F.linear(x, weight)
=======
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        transformed_rows = [
            F.pad(
                coordinates,
                (2 if row_index == self.rotation_gauge_row else 1, 0),
            )
            for row_index, coordinates in enumerate(self.coordinates)
        ]
        weight = self._householder(torch.stack(transformed_rows, dim=0))
        return F.linear(x, weight)
>>>>>>> REPLACE

<<<<<<< SEARCH
            padded = torch.cat((parameter.new_zeros(1), parameter))
            ambient = self._householder(padded, reflector, norm_sq)
            grad_padded = torch.cat(
                (parameter.grad.new_zeros(1), parameter.grad)
            )
=======
            fixed_coordinates = reflector.numel() - parameter.numel()
            padded = torch.cat(
                (parameter.new_zeros(fixed_coordinates), parameter)
            )
            ambient = self._householder(padded, reflector, norm_sq)
            grad_padded = torch.cat(
                (
                    parameter.grad.new_zeros(fixed_coordinates),
                    parameter.grad,
                )
            )
>>>>>>> REPLACE

<<<<<<< SEARCH
            projected = self._householder(
                ambient, reflector, norm_sq
            )
            parameter.copy_(projected[1:])
=======
            projected = self._householder(
                ambient, reflector, norm_sq
            )
            parameter.copy_(projected[fixed_coordinates:])
>>>>>>> REPLACE