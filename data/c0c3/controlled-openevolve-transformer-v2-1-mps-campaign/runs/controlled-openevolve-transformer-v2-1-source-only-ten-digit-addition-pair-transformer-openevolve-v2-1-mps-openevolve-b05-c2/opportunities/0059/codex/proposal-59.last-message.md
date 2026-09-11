MECHANISM: Fourth token-position shift gauge atop the qualified seven-value attention gauge

HYPOTHESIS: Extending the 1520-parameter qualified design with one additional exact token/position common-shift gauge will produce a 1519-parameter model with at least 99% accuracy.

INTENDED_EDIT: Reproduce the qualified seven-value QKV parameterization, then fix a fourth mean-zero coordinate of the anchor token embedding and compensate it in the positional embedding.

EVIDENCE: The seven-value design achieved 99.96% accuracy at 1520 parameters. Using a distinct exact embedding symmetry avoids extending the value-rotation chains whose less-supported anchor directions previously caused large accuracy losses.

<<<<<<< SEARCH
class GaugeFixedEmbedding(nn.Module):
    """Tied embedding with global and three token/position gauges fixed."""
=======
class GaugeFixedEmbedding(nn.Module):
    """Tied embedding with global and four token/position gauges fixed."""
>>>>>>> REPLACE

<<<<<<< SEARCH
        shift = last_coords[-3:]
        gauged_weight = gauged_weight - basis[:, -3:] @ shift
        self.weight_rows = nn.Parameter(gauged_weight[:-1])
        self.last_weight = nn.Parameter(last_coords[:-3])
=======
        shift = last_coords[-4:]
        gauged_weight = gauged_weight - basis[:, -4:] @ shift
        self.weight_rows = nn.Parameter(gauged_weight[:-1])
        self.last_weight = nn.Parameter(last_coords[:-4])
>>>>>>> REPLACE

<<<<<<< SEARCH
        shift = last_coords[-3:]
        gauged_weight = gauged_weight - self.basis[:, -3:] @ shift
        with torch.no_grad():
            self.weight_rows.copy_(gauged_weight[:-1])
            self.last_weight.copy_(last_coords[:-3])
            self.initial_position_shift.copy_(shift)
=======
        shift = last_coords[-4:]
        gauged_weight = gauged_weight - self.basis[:, -4:] @ shift
        with torch.no_grad():
            self.weight_rows.copy_(gauged_weight[:-1])
            self.last_weight.copy_(last_coords[:-4])
            self.initial_position_shift.copy_(shift)
>>>>>>> REPLACE

<<<<<<< SEARCH
        last_row = self.basis @ F.pad(self.last_weight, (0, 3))
=======
        last_row = self.basis @ F.pad(self.last_weight, (0, 4))
>>>>>>> REPLACE

<<<<<<< SEARCH
class ElevenRotationFourValueGaugeFixedQKV(nn.Module):
    """QKV map with eleven query-key and four value-output gauges fixed."""
=======
class ElevenRotationSevenValueGaugeFixedQKV(nn.Module):
    """QKV map with eleven query-key and seven value-output gauges fixed."""
>>>>>>> REPLACE

<<<<<<< SEARCH
        (
            fixed_weight,
            first_value_rotation,
            second_value_rotation,
            first_value_stabilizer_rotation,
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
            "initial_head_two_value_rotation",
            head_two_value_rotation.detach().clone(),
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
            head_two_value_complement_rotation,
            head_two_value_residual_rotation,
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
        self.register_buffer(
            "initial_head_two_value_complement_rotation",
            head_two_value_complement_rotation.detach().clone(),
            persistent=False,
        )
        self.register_buffer(
            "initial_head_two_value_residual_rotation",
            head_two_value_residual_rotation.detach().clone(),
            persistent=False,
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.middle_value_weight = nn.Parameter(
            fixed_weight[value_start + 2:value_start + self.second_query]
        )
        self.head_two_value_weight = nn.Parameter(
            fixed_weight[value_start + self.second_query, 1:]
        )
        self.trailing_weight = nn.Parameter(
            fixed_weight[value_start + self.second_query + 1:-1]
        )
        self.last_weight = nn.Parameter(fixed_weight[-1] @ basis)
=======
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
        residual_row = value_start + self.second_query + 1
        self.head_two_value_residual_weight = nn.Parameter(
            torch.cat(
                (
                    fixed_weight[residual_row, :2],
                    fixed_weight[residual_row, 3:],
                )
            )
        )
        complement_row = value_start + self.second_query + 2
        self.head_two_value_complement_weight = nn.Parameter(
            torch.cat(
                (
                    fixed_weight[complement_row, :1],
                    fixed_weight[complement_row, 2:],
                )
            )
        )
        self.last_weight = nn.Parameter(fixed_weight[-1] @ basis)
>>>>>>> REPLACE

<<<<<<< SEARCH
    def _gauge_fix(
        self, full_weight: torch.Tensor
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
=======
    def _gauge_fix(
        self, full_weight: torch.Tensor
    ) -> Tuple[torch.Tensor, ...]:
>>>>>>> REPLACE

<<<<<<< SEARCH
        value_rotations = []
        value_start = 2 * self.in_features
        for value_offset, input_coord in (
            (0, 0),
            (1, 0),
            (0, 1),
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
        )
=======
        value_rotations = []
        value_start = 2 * self.in_features
        for value_offset, input_coord in (
            (0, 0),
            (1, 0),
            (0, 1),
            (2, 1),
            (self.second_query, 0),
            (self.second_query + 2, 1),
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

        residual_rows = [
            value_start + self.second_query + 1,
            value_start + self.second_query + 3,
        ]
        pivot = fixed_weight[residual_rows, 2]
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
        fixed_weight[residual_rows] = (
            rotation @ fixed_weight[residual_rows]
        )
        fixed_weight[residual_rows[0], 2] = 0.0
        value_rotations.append(rotation)

        return (
            fixed_weight,
            value_rotations[0],
            value_rotations[1],
            value_rotations[2],
            value_rotations[3],
            value_rotations[4],
            value_rotations[5],
            value_rotations[6],
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
        (
            fixed_weight,
            first_value_rotation,
            second_value_rotation,
            first_value_stabilizer_rotation,
            head_two_value_rotation,
        ) = self._gauge_fix(full_weight)
=======
        (
            fixed_weight,
            first_value_rotation,
            second_value_rotation,
            first_value_stabilizer_rotation,
            first_value_complement_rotation,
            head_two_value_rotation,
            head_two_value_complement_rotation,
            head_two_value_residual_rotation,
        ) = self._gauge_fix(full_weight)
>>>>>>> REPLACE

<<<<<<< SEARCH
            self.middle_value_weight.copy_(
                fixed_weight[value_start + 2:value_start + self.second_query]
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
            self.initial_head_two_value_rotation.copy_(
                head_two_value_rotation
            )
=======
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
            residual_row = value_start + self.second_query + 1
            self.head_two_value_residual_weight.copy_(
                torch.cat(
                    (
                        fixed_weight[residual_row, :2],
                        fixed_weight[residual_row, 3:],
                    )
                )
            )
            complement_row = value_start + self.second_query + 2
            self.head_two_value_complement_weight.copy_(
                torch.cat(
                    (
                        fixed_weight[complement_row, :1],
                        fixed_weight[complement_row, 2:],
                    )
                )
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
            self.initial_head_two_value_complement_rotation.copy_(
                head_two_value_complement_rotation
            )
            self.initial_head_two_value_residual_rotation.copy_(
                head_two_value_residual_rotation
            )
>>>>>>> REPLACE

<<<<<<< SEARCH
        first_value_row = F.pad(self.first_value_weight, (2, 0))
        second_value_row = F.pad(self.second_value_weight, (1, 0))
        head_two_value_row = F.pad(self.head_two_value_weight, (1, 0))
        last_row = self.basis @ self.last_weight
        return torch.cat(
            (
                first_row.unsqueeze(0),
                second_row.unsqueeze(0),
                third_row.unsqueeze(0),
                self.first_head_tail,
                head_two_row.unsqueeze(0),
                head_two_second_row.unsqueeze(0),
                head_two_third_row.unsqueeze(0),
                self.pre_value_weight,
                first_value_row.unsqueeze(0),
                second_value_row.unsqueeze(0),
                self.middle_value_weight,
                head_two_value_row.unsqueeze(0),
                self.trailing_weight,
                last_row.unsqueeze(0),
            ),
            dim=0,
        )
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
        head_two_value_residual_row = torch.cat(
            (
                self.head_two_value_residual_weight[:2],
                self.head_two_value_residual_weight.new_zeros(1),
                self.head_two_value_residual_weight[2:],
            )
        )
        head_two_value_complement_row = torch.cat(
            (
                self.head_two_value_complement_weight[:1],
                self.head_two_value_complement_weight.new_zeros(1),
                self.head_two_value_complement_weight[1:],
            )
        )
        last_row = self.basis @ self.last_weight
        return torch.cat(
            (
                first_row.unsqueeze(0),
                second_row.unsqueeze(0),
                third_row.unsqueeze(0),
                self.first_head_tail,
                head_two_row.unsqueeze(0),
                head_two_second_row.unsqueeze(0),
                head_two_third_row.unsqueeze(0),
                self.pre_value_weight,
                first_value_row.unsqueeze(0),
                second_value_row.unsqueeze(0),
                third_value_row.unsqueeze(0),
                self.first_head_value_tail,
                head_two_value_row.unsqueeze(0),
                head_two_value_residual_row.unsqueeze(0),
                head_two_value_complement_row.unsqueeze(0),
                last_row.unsqueeze(0),
            ),
            dim=0,
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.qkv = ElevenRotationFourValueGaugeFixedQKV(
            d_model, 3 * d_model, self.head_dim
        )
=======
        self.qkv = ElevenRotationSevenValueGaugeFixedQKV(
            d_model, 3 * d_model, self.head_dim
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
            stabilized_columns = (
                self.proj.weight[:, :2]
                @ self.qkv.initial_first_value_stabilizer_rotation.transpose(
                    0, 1
                )
            )
            self.proj.weight[:, :2].copy_(stabilized_columns)
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
=======
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
            head_two_complement_columns = (
                self.proj.weight[
                    :, head_two_start + 2:head_two_start + 4
                ]
                @ self.qkv.initial_head_two_value_complement_rotation.transpose(
                    0, 1
                )
            )
            self.proj.weight[
                :, head_two_start + 2:head_two_start + 4
            ].copy_(head_two_complement_columns)
            residual_indices = [
                head_two_start + 1,
                head_two_start + 3,
            ]
            residual_columns = (
                self.proj.weight[:, residual_indices]
                @ self.qkv.initial_head_two_value_residual_rotation.transpose(
                    0, 1
                )
            )
            self.proj.weight[:, residual_indices] = residual_columns
>>>>>>> REPLACE

<<<<<<< SEARCH
        elif isinstance(module, ElevenRotationFourValueGaugeFixedQKV):
=======
        elif isinstance(module, ElevenRotationSevenValueGaugeFixedQKV):
>>>>>>> REPLACE

<<<<<<< SEARCH
            self.pos_emb.weight[:, -3:].add_(
                self.token_emb.initial_position_shift
            )
=======
            self.pos_emb.weight[:, -4:].add_(
                self.token_emb.initial_position_shift
            )
>>>>>>> REPLACE