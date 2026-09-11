MECHANISM: Residual first-head value-output stabilizer

HYPOTHESIS: Adding a column-one stabilizer rotation within the already column-zero-anchored first-head value subspace to the qualified 1524-parameter mirrored-head design will produce a 1523-parameter model with at least 99% accuracy.

INTENDED_EDIT: Retain the two qualified first-head value rotations, add the qualified first second-head rotation, then rotate first-head value channels 0–1 on input column 1 and omit the resulting fixed coefficient while inversely rotating the matching projection columns.

EVIDENCE: The mirrored first second-head rotation achieved 99.72% at 1524 parameters, whereas extending either head sequentially into another channel failed; the proposed residual rotation acts only within the two first-head rows whose column-zero coefficients are already fixed, testing a distinct stabilizer gauge analogous to the successful query-key stabilizers.

<<<<<<< SEARCH
class ElevenRotationTwoValueGaugeFixedQKV(nn.Module):
    """QKV map with eleven query-key and two value-output gauges fixed."""
    def __init__(self, in_features: int, out_features: int, head_dim: int):
        super().__init__()
        if out_features != 3 * in_features:
            raise ValueError("gauge-fixed QKV requires out_features == 3 * in_features")
        if head_dim < 4 or 2 * head_dim > in_features:
            raise ValueError("gauge fixing requires two suitable heads")
        self.in_features = in_features
        self.out_features = out_features
        self.second_query = head_dim
        basis = _mean_zero_basis(in_features)
        self.register_buffer("basis", basis, persistent=False)

        full_weight = torch.empty(out_features, in_features)
        nn.init.kaiming_uniform_(full_weight, a=math.sqrt(5))
        discarded_bias = torch.empty(out_features)
        bound = 1.0 / math.sqrt(in_features)
        nn.init.uniform_(discarded_bias, -bound, bound)

        (
            fixed_weight,
            first_value_rotation,
            second_value_rotation,
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
=======
class ElevenRotationFourValueGaugeFixedQKV(nn.Module):
    """QKV map with eleven query-key and four value-output gauges fixed."""
    def __init__(self, in_features: int, out_features: int, head_dim: int):
        super().__init__()
        if out_features != 3 * in_features:
            raise ValueError("gauge-fixed QKV requires out_features == 3 * in_features")
        if head_dim < 4 or 2 * head_dim > in_features:
            raise ValueError("gauge fixing requires two suitable heads")
        self.in_features = in_features
        self.out_features = out_features
        self.second_query = head_dim
        basis = _mean_zero_basis(in_features)
        self.register_buffer("basis", basis, persistent=False)

        full_weight = torch.empty(out_features, in_features)
        nn.init.kaiming_uniform_(full_weight, a=math.sqrt(5))
        discarded_bias = torch.empty(out_features)
        bound = 1.0 / math.sqrt(in_features)
        nn.init.uniform_(discarded_bias, -bound, bound)

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
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.first_value_weight = nn.Parameter(
            fixed_weight[2 * self.in_features, 1:]
        )
        self.second_value_weight = nn.Parameter(
            fixed_weight[2 * self.in_features + 1, 1:]
        )
        self.trailing_weight = nn.Parameter(
            fixed_weight[2 * self.in_features + 2:-1]
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
>>>>>>> REPLACE

<<<<<<< SEARCH
    def _gauge_fix(
        self, full_weight: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
=======
    def _gauge_fix(
        self, full_weight: torch.Tensor
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
>>>>>>> REPLACE

<<<<<<< SEARCH
        value_rotations = []
        value_start = 2 * self.in_features
        for value_offset in (0, 1):
            row_start = value_start + value_offset
            pivot = fixed_weight[row_start:row_start + 2, 0]
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
            fixed_weight[row_start, 0] = 0.0
            value_rotations.append(rotation)

        return fixed_weight, value_rotations[0], value_rotations[1]
=======
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
>>>>>>> REPLACE

<<<<<<< SEARCH
        (
            fixed_weight,
            first_value_rotation,
            second_value_rotation,
        ) = self._gauge_fix(full_weight)
=======
        (
            fixed_weight,
            first_value_rotation,
            second_value_rotation,
            first_value_stabilizer_rotation,
            head_two_value_rotation,
        ) = self._gauge_fix(full_weight)
>>>>>>> REPLACE

<<<<<<< SEARCH
            self.first_value_weight.copy_(
                fixed_weight[2 * self.in_features, 1:]
            )
            self.second_value_weight.copy_(
                fixed_weight[2 * self.in_features + 1, 1:]
            )
            self.trailing_weight.copy_(
                fixed_weight[2 * self.in_features + 2:-1]
            )
            self.last_weight.copy_(fixed_weight[-1] @ self.basis)
            self.initial_first_value_rotation.copy_(first_value_rotation)
            self.initial_second_value_rotation.copy_(second_value_rotation)
=======
            value_start = 2 * self.in_features
            self.first_value_weight.copy_(
                fixed_weight[value_start, 2:]
            )
            self.second_value_weight.copy_(
                fixed_weight[value_start + 1, 1:]
            )
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
>>>>>>> REPLACE

<<<<<<< SEARCH
        first_value_row = F.pad(self.first_value_weight, (1, 0))
        second_value_row = F.pad(self.second_value_weight, (1, 0))
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
                self.trailing_weight,
                last_row.unsqueeze(0),
            ),
            dim=0,
        )
=======
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
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.qkv = ElevenRotationTwoValueGaugeFixedQKV(
            d_model, 3 * d_model, self.head_dim
        )
=======
        self.qkv = ElevenRotationFourValueGaugeFixedQKV(
            d_model, 3 * d_model, self.head_dim
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
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
        elif isinstance(module, ElevenRotationTwoValueGaugeFixedQKV):
=======
        elif isinstance(module, ElevenRotationFourValueGaugeFixedQKV):
>>>>>>> REPLACE