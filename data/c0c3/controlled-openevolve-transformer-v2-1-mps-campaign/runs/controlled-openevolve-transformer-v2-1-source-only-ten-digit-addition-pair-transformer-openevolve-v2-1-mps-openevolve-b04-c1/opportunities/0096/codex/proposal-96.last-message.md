MECHANISM: Unbiased-head Q/K shear gauge fixing

HYPOTHESIS: Fixing a second query-weight coordinate in the unbiased attention head through a compensating Q/K shear will reduce the model from 870 to 869 learned parameters while retaining at least 99% accuracy, because the inverse-transpose key transformation preserves every attention logit.

INTENDED_EDIT: Triangularize two coordinates of the unbiased head’s query rows, omit the resulting zero coordinate, and generalize ambient AdamW to optimize noncontiguous gauge coordinates.

EVIDENCE: The preceding joint Q/K rotational gauge retained 99.90% accuracy at 870 parameters; a shear is another exact dimension of the same Q/K change-of-basis invariance and removes no attention function capacity.

<<<<<<< SEARCH
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

        inv_sqrt = d_model ** -0.5
=======
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

        inv_sqrt = d_model ** -0.5
>>>>>>> REPLACE

<<<<<<< SEARCH
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
=======
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
            transformed[q1, 1] = 0.0

            # An upper-triangular Q shear preserves the rotation zero. Choose
            # a stable pivot in the second Q row and apply the inverse-
            # transpose shear to K, leaving every Q/K dot product unchanged.
            pivot = int(
                torch.argmax(transformed[q1, 2:].abs()).item()
            ) + 2
            self.shear_pivot.fill_(pivot)
            shear = transformed[q0, pivot] / transformed[q1, pivot]
            transformed[q0, 1:] = (
                transformed[q0, 1:] - shear * transformed[q1, 1:]
            )
            transformed[k1, 1:] = (
                transformed[k1, 1:] + shear * transformed[k0, 1:]
            )

            for row_index, (coordinates, row) in enumerate(
                zip(self.coordinates, transformed)
            ):
                if row_index == self.shear_gauge_row:
                    coordinates.copy_(
                        torch.cat((row[1:pivot], row[pivot + 1 :]))
                    )
                elif row_index == self.rotation_gauge_row:
                    coordinates.copy_(row[2:])
                else:
                    coordinates.copy_(row[1:])
>>>>>>> REPLACE

<<<<<<< SEARCH
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
=======
    def ambient_coordinate_indices(
        self, row_index: int
    ) -> Tuple[int, ...]:
        if row_index == self.shear_gauge_row:
            pivot = int(self.shear_pivot.item())
            return tuple(
                index
                for index in range(1, self.d_model)
                if index != pivot
            )
        if row_index == self.rotation_gauge_row:
            return tuple(range(2, self.d_model))
        return tuple(range(1, self.d_model))

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
                row = F.pad(coordinates, (2, 0))
            else:
                row = F.pad(coordinates, (1, 0))
            transformed_rows.append(row)

        weight = self._householder(torch.stack(transformed_rows, dim=0))
        return F.linear(x, weight)
>>>>>>> REPLACE

<<<<<<< SEARCH
        entries: List[Tuple[torch.nn.Parameter, torch.Tensor, float]],
        lr: float,
        weight_decay: float,
    ):
        self.entries = entries
        self.param_groups = [{"lr": lr}]
        self.weight_decay = weight_decay
        self.beta1 = 0.9
        self.beta2 = 0.999
        self.eps = 1e-8
        self.states = [
            {
                "step": 0,
                "exp_avg": torch.zeros_like(reflector),
                "exp_avg_sq": torch.zeros_like(reflector),
            }
            for _, reflector, _ in entries
        ]
=======
        entries: List[Tuple],
        lr: float,
        weight_decay: float,
    ):
        self.entries = entries
        self.param_groups = [{"lr": lr}]
        self.weight_decay = weight_decay
        self.beta1 = 0.9
        self.beta2 = 0.999
        self.eps = 1e-8
        self.states = [
            {
                "step": 0,
                "exp_avg": torch.zeros_like(entry[1]),
                "exp_avg_sq": torch.zeros_like(entry[1]),
            }
            for entry in entries
        ]
>>>>>>> REPLACE

<<<<<<< SEARCH
    def zero_grad(self, set_to_none: bool = True) -> None:
        for parameter, _, _ in self.entries:
            if parameter.grad is not None:
                if set_to_none:
                    parameter.grad = None
                else:
                    parameter.grad.zero_()

    @torch.no_grad()
    def step(self) -> None:
        lr = self.param_groups[0]["lr"]
        for (parameter, reflector, norm_sq), state in zip(
            self.entries, self.states
        ):
            if parameter.grad is None:
                continue

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
            ambient_grad = self._householder(
                grad_padded, reflector, norm_sq
            )

            state["step"] += 1
            exp_avg = state["exp_avg"]
            exp_avg_sq = state["exp_avg_sq"]
            exp_avg.mul_(self.beta1).add_(
                ambient_grad, alpha=1.0 - self.beta1
            )
            exp_avg_sq.mul_(self.beta2).addcmul_(
                ambient_grad, ambient_grad, value=1.0 - self.beta2
            )

            ambient.mul_(1.0 - lr * self.weight_decay)
            bias_correction1 = 1.0 - self.beta1 ** state["step"]
            bias_correction2 = 1.0 - self.beta2 ** state["step"]
            denom = (
                exp_avg_sq.sqrt()
                .div(math.sqrt(bias_correction2))
                .add(self.eps)
            )
            ambient.addcdiv_(
                exp_avg, denom, value=-lr / bias_correction1
            )

            projected = self._householder(
                ambient, reflector, norm_sq
            )
            parameter.copy_(projected[fixed_coordinates:])
=======
    def zero_grad(self, set_to_none: bool = True) -> None:
        for entry in self.entries:
            parameter = entry[0]
            if parameter.grad is not None:
                if set_to_none:
                    parameter.grad = None
                else:
                    parameter.grad.zero_()

    @torch.no_grad()
    def step(self) -> None:
        lr = self.param_groups[0]["lr"]
        for entry, state in zip(self.entries, self.states):
            parameter, reflector, norm_sq = entry[:3]
            if parameter.grad is None:
                continue

            if len(entry) == 3:
                fixed_coordinates = reflector.numel() - parameter.numel()
                padded = torch.cat(
                    (parameter.new_zeros(fixed_coordinates), parameter)
                )
                grad_padded = torch.cat(
                    (
                        parameter.grad.new_zeros(fixed_coordinates),
                        parameter.grad,
                    )
                )
            else:
                coordinate_indices = list(entry[3])
                padded = parameter.new_zeros(reflector.numel())
                padded[coordinate_indices] = parameter
                grad_padded = parameter.grad.new_zeros(reflector.numel())
                grad_padded[coordinate_indices] = parameter.grad

            ambient = self._householder(padded, reflector, norm_sq)
            ambient_grad = self._householder(
                grad_padded, reflector, norm_sq
            )

            state["step"] += 1
            exp_avg = state["exp_avg"]
            exp_avg_sq = state["exp_avg_sq"]
            exp_avg.mul_(self.beta1).add_(
                ambient_grad, alpha=1.0 - self.beta1
            )
            exp_avg_sq.mul_(self.beta2).addcmul_(
                ambient_grad, ambient_grad, value=1.0 - self.beta2
            )

            ambient.mul_(1.0 - lr * self.weight_decay)
            bias_correction1 = 1.0 - self.beta1 ** state["step"]
            bias_correction2 = 1.0 - self.beta2 ** state["step"]
            denom = (
                exp_avg_sq.sqrt()
                .div(math.sqrt(bias_correction2))
                .add(self.eps)
            )
            ambient.addcdiv_(
                exp_avg, denom, value=-lr / bias_correction1
            )

            projected = self._householder(
                ambient, reflector, norm_sq
            )
            if len(entry) == 3:
                parameter.copy_(projected[fixed_coordinates:])
            else:
                parameter.copy_(projected[coordinate_indices])
>>>>>>> REPLACE

<<<<<<< SEARCH
    ] + [
        (
            coordinates,
            block.attn.qkv.reflector,
            block.attn.qkv.reflector_norm_sq,
        )
        for block in model.blocks
        for coordinates in block.attn.qkv.coordinates
    ]
    gauge_ids = {id(parameter) for parameter, _, _ in gauge_entries}
=======
    ] + [
        (
            coordinates,
            block.attn.qkv.reflector,
            block.attn.qkv.reflector_norm_sq,
            block.attn.qkv.ambient_coordinate_indices(row_index),
        )
        for block in model.blocks
        for row_index, coordinates in enumerate(
            block.attn.qkv.coordinates
        )
    ]
    gauge_ids = {id(entry[0]) for entry in gauge_entries}
>>>>>>> REPLACE