MECHANISM: Complement-subspace value-output orthogonal gauge fixing

HYPOTHESIS: Rotating first-head value channels 2–3 on input column 1 will produce a 1522-parameter model with at least 99% accuracy while preserving the initialized model function and all four qualified value anchors.

INTENDED_EDIT: Reproduce the qualified 1523-parameter four-value-gauge design, add an orthogonal rotation confined to first-head value rows 2–3, omit the resulting fixed `value_weight[2, 1]` coefficient, and inversely rotate projection columns 2–3.

EVIDENCE: The four-value-gauge design achieved 99.57% accuracy at 1523 parameters. Additional column-zero chain rotations failed, whereas the qualified column-one stabilizer showed that changing the gauge direction can retain trainability; this complementary rotation preserves every existing zero without extending either failed column-zero chain.

<<<<<<< SEARCH
class ElevenRotationGaugeFixedQKV(nn.Module):
    """QKV map with input, query-key, and one value-output gauge fixed."""
=======
class ElevenRotationFiveValueGaugeFixedQKV(nn.Module):
    """QKV map with eleven query-key and five value-output gauges fixed."""
>>>>>>> REPLACE

<<<<<<< SEARCH
        fixed_weight, value_rotation = self._gauge_fix(full_weight)
        self.register_buffer(
            "initial_value_rotation",
            value_rotation.detach().clone(),
            persistent=False,
        )
=======
        (
            fixed_weight,
            first_value_rotation,
            second_value_rotation,
            first_value_stabilizer_rotation,
            first_value_complement_rotation,
            head_two_value_rotation,
        ) = self._gauge_fix(full_weight)
        self.register_buffer(
            "initial_first_value_rotation",
            first_value_rotation.detach().clone(),
            persistent=False,
        )
        self.register_buffer(
            "initial_second_value_rotation",
            second_value_rotation.detach().clone(),
            persistent=False,
        )
        self.register_buffer(
            "initial_first_value_stabilizer_rotation",
            first_value_stabilizer_rotation.detach().clone(),
            persistent=False,
        )
        self.register_buffer(
            "initial_first_value_complement_rotation",
            first_value_complement_rotation.detach().clone(),
            persistent=False,
        )
        self.register_buffer(
            "initial_head_two_value_rotation",
            head_two_value_rotation.detach().clone(),
            persistent=False,
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.first_value_weight = nn.Parameter(
            fixed_weight[2 * self.in_features, 1:]
        )
        self.trailing_weight = nn.Parameter(
            fixed_weight[2 * self.in_features + 1:-1]
        )
        self.last_weight = nn.Parameter(fixed_weight[-1] @ basis)
=======
        value_start = 2 * self.in_features
        self.first_value_weight = nn.Parameter(
            fixed_weight[value_start, 2:]
        )
        self.second_value_weight = nn.Parameter(
            fixed_weight[value_start + 1, 1:]
        )
        self.third_value_weight = nn.Parameter(
            torch.cat(
                (
                    fixed_weight[value_start + 2, :1],
                    fixed_weight[value_start + 2, 2:],
                )
            )
        )
        self.first_head_value_tail = nn.Parameter(
            fixed_weight[value_start + 3:value_start + self.second_query]
        )
        self.head_two_value_weight = nn.Parameter(
            fixed_weight[value_start + self.second_query, 1:]
        )
        self.trailing_weight = nn.Parameter(
            fixed_weight[value_start + self.second_query + 1:-1]
        )
        self.last_weight = nn.Parameter(fixed_weight[-1] @ basis)
>>>>>>> REPLACE

<<<<<<< SEARCH
    def _gauge_fix(
        self, full_weight: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
=======
    def _gauge_fix(
        self, full_weight: torch.Tensor
    ) -> Tuple[torch.Tensor, ...]:
>>>>>>> REPLACE

<<<<<<< SEARCH
        value_start = 2 * self.in_features
        pivot = fixed_weight[value_start:value_start + 2, 0]
        radius = pivot.norm().clamp_min(
            torch.finfo(full_weight.dtype).tiny
        )
        cosine = pivot[1] / radius
        sine = -pivot[0] / radius
        value_rotation = torch.stack(
            (
                torch.stack((cosine, sine)),
                torch.stack((-sine, cosine)),
            )
        )
        fixed_weight[value_start:value_start + 2] = (
            value_rotation @ fixed_weight[value_start:value_start + 2]
        )
        fixed_weight[value_start, 0] = 0.0
        return fixed_weight, value_rotation
=======
        value_rotations = []
        value_start = 2 * self.in_features
        for value_offset, input_coord in (
            (0, 0),
            (1, 0),
            (0, 1),
            (2, 1),
            (self.second_query, 0),
        ):
            row_start = value_start + value_offset
            pivot = fixed_weight[
                row_start:row_start + 2, input_coord
            ]
            radius = pivot.norm().clamp_min(
                torch.finfo(full_weight.dtype).tiny
            )
            cosine = pivot[1] / radius
            sine = -pivot[0] / radius
            rotation = torch.stack(
                (
                    torch.stack((cosine, sine)),
                    torch.stack((-sine, cosine)),
                )
            )
            fixed_weight[row_start:row_start + 2] = (
                rotation @ fixed_weight[row_start:row_start + 2]
            )
            fixed_weight[row_start, input_coord] = 0.0
            value_rotations.append(rotation)

        return (
            fixed_weight,
            value_rotations[0],
            value_rotations[1],
            value_rotations[2],
            value_rotations[3],
            value_rotations[4],
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
        fixed_weight, value_rotation = self._gauge_fix(full_weight)
        with torch.no_grad():
=======
        (
            fixed_weight,
            first_value_rotation,
            second_value_rotation,
            first_value_stabilizer_rotation,
            first_value_complement_rotation,
            head_two_value_rotation,
        ) = self._gauge_fix(full_weight)
        with torch.no_grad():
>>>>>>> REPLACE

<<<<<<< SEARCH
            self.first_value_weight.copy_(
                fixed_weight[2 * self.in_features, 1:]
            )
            self.trailing_weight.copy_(
                fixed_weight[2 * self.in_features + 1:-1]
            )
            self.last_weight.copy_(fixed_weight[-1] @ self.basis)
            self.initial_value_rotation.copy_(value_rotation)
=======
            value_start = 2 * self.in_features
            self.first_value_weight.copy_(
                fixed_weight[value_start, 2:]
            )
            self.second_value_weight.copy_(
                fixed_weight[value_start + 1, 1:]
            )
            self.third_value_weight.copy_(
                torch.cat(
                    (
                        fixed_weight[value_start + 2, :1],
                        fixed_weight[value_start + 2, 2:],
                    )
                )
            )
            self.first_head_value_tail.copy_(
                fixed_weight[
                    value_start + 3:value_start + self.second_query
                ]
            )
            self.head_two_value_weight.copy_(
                fixed_weight[value_start + self.second_query, 1:]
            )
            self.trailing_weight.copy_(
                fixed_weight[value_start + self.second_query + 1:-1]
            )
            self.last_weight.copy_(fixed_weight[-1] @ self.basis)
            self.initial_first_value_rotation.copy_(first_value_rotation)
            self.initial_second_value_rotation.copy_(second_value_rotation)
            self.initial_first_value_stabilizer_rotation.copy_(
                first_value_stabilizer_rotation
            )
            self.initial_first_value_complement_rotation.copy_(
                first_value_complement_rotation
            )
            self.initial_head_two_value_rotation.copy_(
                head_two_value_rotation
            )
>>>>>>> REPLACE

<<<<<<< SEARCH
        first_value_row = F.pad(self.first_value_weight, (1, 0))
        last_row = self.basis @ self.last_weight
=======
        first_value_row = F.pad(self.first_value_weight, (2, 0))
        second_value_row = F.pad(self.second_value_weight, (1, 0))
        third_value_row = torch.cat(
            (
                self.third_value_weight[:1],
                self.third_value_weight.new_zeros(1),
                self.third_value_weight[1:],
            )
        )
        head_two_value_row = F.pad(self.head_two_value_weight, (1, 0))
        last_row = self.basis @ self.last_weight
>>>>>>> REPLACE

<<<<<<< SEARCH
                self.pre_value_weight,
                first_value_row.unsqueeze(0),
                self.trailing_weight,
                last_row.unsqueeze(0),
=======
                self.pre_value_weight,
                first_value_row.unsqueeze(0),
                second_value_row.unsqueeze(0),
                third_value_row.unsqueeze(0),
                self.first_head_value_tail,
                head_two_value_row.unsqueeze(0),
                self.trailing_weight,
                last_row.unsqueeze(0),
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.qkv = ElevenRotationGaugeFixedQKV(
            d_model, 3 * d_model, self.head_dim
        )
=======
        self.qkv = ElevenRotationFiveValueGaugeFixedQKV(
            d_model, 3 * d_model, self.head_dim
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
    def apply_initial_value_rotation(self) -> None:
        with torch.no_grad():
            rotated_columns = (
                self.proj.weight[:, :2]
                @ self.qkv.initial_value_rotation.transpose(0, 1)
            )
            self.proj.weight[:, :2].copy_(rotated_columns)
=======
    def apply_initial_value_rotations(self) -> None:
        with torch.no_grad():
            first_columns = (
                self.proj.weight[:, :2]
                @ self.qkv.initial_first_value_rotation.transpose(0, 1)
            )
            self.proj.weight[:, :2].copy_(first_columns)
            second_columns = (
                self.proj.weight[:, 1:3]
                @ self.qkv.initial_second_value_rotation.transpose(0, 1)
            )
            self.proj.weight[:, 1:3].copy_(second_columns)
            stabilized_columns = (
                self.proj.weight[:, :2]
                @ self.qkv.initial_first_value_stabilizer_rotation.transpose(
                    0, 1
                )
            )
            self.proj.weight[:, :2].copy_(stabilized_columns)
            complement_columns = (
                self.proj.weight[:, 2:4]
                @ self.qkv.initial_first_value_complement_rotation.transpose(
                    0, 1
                )
            )
            self.proj.weight[:, 2:4].copy_(complement_columns)
            head_two_start = self.head_dim
            head_two_columns = (
                self.proj.weight[
                    :, head_two_start:head_two_start + 2
                ]
                @ self.qkv.initial_head_two_value_rotation.transpose(0, 1)
            )
            self.proj.weight[
                :, head_two_start:head_two_start + 2
            ].copy_(head_two_columns)
>>>>>>> REPLACE

<<<<<<< SEARCH
            block.attn.apply_initial_value_rotation()
=======
            block.attn.apply_initial_value_rotations()
>>>>>>> REPLACE

<<<<<<< SEARCH
        elif isinstance(module, ElevenRotationGaugeFixedQKV):
=======
        elif isinstance(module, ElevenRotationFiveValueGaugeFixedQKV):
>>>>>>> REPLACE