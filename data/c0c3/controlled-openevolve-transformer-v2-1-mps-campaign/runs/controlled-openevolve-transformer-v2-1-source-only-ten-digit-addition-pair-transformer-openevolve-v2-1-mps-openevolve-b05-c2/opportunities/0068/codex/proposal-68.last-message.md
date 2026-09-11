MECHANISM: Complete LayerNorm scale gauge

HYPOTHESIS: Reproducing the qualified 1512-parameter design and fixing the eighth and final `ln1` scale will yield a 1511-parameter model with at least 99% accuracy.

INTENDED_EDIT: Apply the qualified complete embedding, seventh value-output, and query-bias gauges, then make all eight pre-attention LayerNorm scales fixed units.

EVIDENCE: The seven-fixed-scale design achieved 99.96% accuracy at 1512 parameters, after the five- and six-fixed-scale designs achieved 99.84% and 99.91%; fixing the remaining scale is the closest supported reduction.

<<<<<<< SEARCH
class GaugeFixedEmbedding(nn.Module):
    """Tied embedding with global and three token/position gauges fixed."""
    def __init__(self, num_embeddings: int, embedding_dim: int):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        basis = _mean_zero_basis(embedding_dim)
        self.register_buffer("basis", basis, persistent=False)

        full_weight = torch.empty(num_embeddings, embedding_dim)
        nn.init.normal_(full_weight)
        anchor = full_weight[-1].mean()
        gauged_weight = full_weight - anchor
        last_coords = gauged_weight[-1] @ basis
        shift = last_coords[-3:]
        gauged_weight = gauged_weight - basis[:, -3:] @ shift
        self.weight_rows = nn.Parameter(gauged_weight[:-1])
        self.last_weight = nn.Parameter(last_coords[:-3])
        self.register_buffer(
            "initial_position_shift", shift.detach().clone(), persistent=False
        )

    def reset_from_full(self, full_weight: torch.Tensor) -> None:
        anchor = full_weight[-1].mean()
        gauged_weight = full_weight - anchor
        last_coords = gauged_weight[-1] @ self.basis
        shift = last_coords[-3:]
        gauged_weight = gauged_weight - self.basis[:, -3:] @ shift
        with torch.no_grad():
            self.weight_rows.copy_(gauged_weight[:-1])
            self.last_weight.copy_(last_coords[:-3])
            self.initial_position_shift.copy_(shift)

    @property
    def weight(self) -> torch.Tensor:
        last_row = self.basis @ F.pad(self.last_weight, (0, 3))
        return torch.cat((self.weight_rows, last_row.unsqueeze(0)), dim=0)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        return F.embedding(idx, self.weight)
=======
class GaugeFixedEmbedding(nn.Module):
    """Tied embedding with every token/position common-shift gauge fixed."""
    def __init__(self, num_embeddings: int, embedding_dim: int):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        basis = _mean_zero_basis(embedding_dim)
        self.register_buffer("basis", basis, persistent=False)

        full_weight = torch.empty(num_embeddings, embedding_dim)
        nn.init.normal_(full_weight)
        anchor = full_weight[-1].mean()
        gauged_weight = full_weight - anchor
        last_coords = gauged_weight[-1] @ basis
        shift = last_coords
        gauged_weight = gauged_weight - basis @ shift
        self.weight_rows = nn.Parameter(gauged_weight[:-1])
        self.register_buffer(
            "initial_position_shift", shift.detach().clone(), persistent=False
        )

    def reset_from_full(self, full_weight: torch.Tensor) -> None:
        anchor = full_weight[-1].mean()
        gauged_weight = full_weight - anchor
        last_coords = gauged_weight[-1] @ self.basis
        shift = last_coords
        gauged_weight = gauged_weight - self.basis @ shift
        with torch.no_grad():
            self.weight_rows.copy_(gauged_weight[:-1])
            self.initial_position_shift.copy_(shift)

    @property
    def weight(self) -> torch.Tensor:
        last_row = self.weight_rows.new_zeros(self.embedding_dim)
        return torch.cat((self.weight_rows, last_row.unsqueeze(0)), dim=0)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        return F.embedding(idx, self.weight)
>>>>>>> REPLACE

<<<<<<< SEARCH
class ElevenRotationSixValueGaugeFixedQKV(nn.Module):
    """QKV map with eleven query-key and six value-output gauges fixed."""
=======
class ElevenRotationSevenValueGaugeFixedQKV(nn.Module):
    """QKV map with eleven query-key and seven value-output gauges fixed."""
>>>>>>> REPLACE

<<<<<<< SEARCH
            head_two_value_rotation,
            head_two_value_complement_rotation,
        ) = self._gauge_fix(full_weight)
        self.register_buffer(
            "initial_first_value_rotation",
=======
            head_two_value_rotation,
            head_two_value_complement_rotation,
            head_two_value_residual_rotation,
        ) = self._gauge_fix(full_weight)
        self.register_buffer(
            "initial_first_value_rotation",
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.register_buffer(
            "initial_head_two_value_complement_rotation",
            head_two_value_complement_rotation.detach().clone(),
            persistent=False,
        )
        self.first_weight = nn.Parameter(fixed_weight[0, 3:])
=======
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
        self.first_weight = nn.Parameter(fixed_weight[0, 3:])
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.head_two_value_middle = nn.Parameter(
            fixed_weight[
                value_start + self.second_query + 1:
                value_start + self.second_query + 2
            ]
        )
=======
        residual_row = value_start + self.second_query + 1
        self.head_two_value_residual_weight = nn.Parameter(
            torch.cat(
                (
                    fixed_weight[residual_row, :2],
                    fixed_weight[residual_row, 3:],
                )
            )
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
            fixed_weight[row_start, input_coord] = 0.0
            value_rotations.append(rotation)

        return (
            fixed_weight,
            value_rotations[0],
            value_rotations[1],
            value_rotations[2],
            value_rotations[3],
            value_rotations[4],
            value_rotations[5],
        )
=======
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
            head_two_value_rotation,
            head_two_value_complement_rotation,
        ) = self._gauge_fix(full_weight)
        with torch.no_grad():
=======
            head_two_value_rotation,
            head_two_value_complement_rotation,
            head_two_value_residual_rotation,
        ) = self._gauge_fix(full_weight)
        with torch.no_grad():
>>>>>>> REPLACE

<<<<<<< SEARCH
            self.head_two_value_middle.copy_(
                fixed_weight[
                    value_start + self.second_query + 1:
                    value_start + self.second_query + 2
                ]
            )
=======
            residual_row = value_start + self.second_query + 1
            self.head_two_value_residual_weight.copy_(
                torch.cat(
                    (
                        fixed_weight[residual_row, :2],
                        fixed_weight[residual_row, 3:],
                    )
                )
            )
>>>>>>> REPLACE

<<<<<<< SEARCH
            self.initial_head_two_value_complement_rotation.copy_(
                head_two_value_complement_rotation
            )

    @property
=======
            self.initial_head_two_value_complement_rotation.copy_(
                head_two_value_complement_rotation
            )
            self.initial_head_two_value_residual_rotation.copy_(
                head_two_value_residual_rotation
            )

    @property
>>>>>>> REPLACE

<<<<<<< SEARCH
        head_two_value_row = F.pad(self.head_two_value_weight, (1, 0))
        head_two_value_complement_row = torch.cat(
=======
        head_two_value_row = F.pad(self.head_two_value_weight, (1, 0))
        head_two_value_residual_row = torch.cat(
            (
                self.head_two_value_residual_weight[:2],
                self.head_two_value_residual_weight.new_zeros(1),
                self.head_two_value_residual_weight[2:],
            )
        )
        head_two_value_complement_row = torch.cat(
>>>>>>> REPLACE

<<<<<<< SEARCH
                head_two_value_row.unsqueeze(0),
                self.head_two_value_middle,
                head_two_value_complement_row.unsqueeze(0),
=======
                head_two_value_row.unsqueeze(0),
                head_two_value_residual_row.unsqueeze(0),
                head_two_value_complement_row.unsqueeze(0),
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.qkv = ElevenRotationSixValueGaugeFixedQKV(
            d_model, 3 * d_model, self.head_dim
        )
        self.q_bias = nn.Parameter(torch.zeros(d_model))
=======
        self.qkv = ElevenRotationSevenValueGaugeFixedQKV(
            d_model, 3 * d_model, self.head_dim
        )
        self.q_bias = nn.Parameter(torch.zeros(d_model - 1))
>>>>>>> REPLACE

<<<<<<< SEARCH
            self.proj.weight[
                :, head_two_start + 2:head_two_start + 4
            ].copy_(head_two_complement_columns)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
=======
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
>>>>>>> REPLACE

<<<<<<< SEARCH
        qkv_bias = torch.cat(
            (
                self.q_bias,
                self.qkv.weight.new_zeros(d_model),
                self.qkv.weight.new_zeros(d_model),
            )
        )
=======
        q_bias = torch.cat(
            (
                self.q_bias[:self.head_dim],
                self.q_bias.new_zeros(1),
                self.q_bias[self.head_dim:],
            )
        )
        qkv_bias = torch.cat(
            (
                q_bias,
                self.qkv.weight.new_zeros(d_model),
                self.qkv.weight.new_zeros(d_model),
            )
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
class QuadrupleAnchoredScaleLayerNorm(nn.Module):
    """LayerNorm with four scales fixed through the following linear-map gauge."""
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim - 4))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = F.pad(self.weight, (0, 4), value=1.0)
        return F.layer_norm(x, (self.dim,), weight, None, self.eps)
=======
class FullyAnchoredScaleLayerNorm(nn.Module):
    """LayerNorm with every scale fixed through the following linear-map gauge."""
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.dim = dim
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.layer_norm(x, (self.dim,), None, None, self.eps)
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.ln1 = QuadrupleAnchoredScaleLayerNorm(cfg.d_model)
=======
        self.ln1 = FullyAnchoredScaleLayerNorm(cfg.d_model)
>>>>>>> REPLACE

<<<<<<< SEARCH
        elif isinstance(module, ElevenRotationSixValueGaugeFixedQKV):
=======
        elif isinstance(module, ElevenRotationSevenValueGaugeFixedQKV):
>>>>>>> REPLACE

<<<<<<< SEARCH
            self.pos_emb.weight[:, -3:].add_(
                self.token_emb.initial_position_shift
            )
=======
            self.pos_emb.weight.add_(
                self.token_emb.initial_position_shift
            )
>>>>>>> REPLACE