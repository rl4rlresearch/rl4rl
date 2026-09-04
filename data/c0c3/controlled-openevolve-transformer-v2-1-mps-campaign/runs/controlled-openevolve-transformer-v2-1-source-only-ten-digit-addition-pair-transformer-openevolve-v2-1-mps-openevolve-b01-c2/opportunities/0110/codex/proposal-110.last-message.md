MECHANISM: Complementary position-zero Helmert contrast quotient

HYPOTHESIS: Removing the final position-zero Helmert contrast `(2 + 4 + 5 - 3*6)` while retaining `(2 + 4 - 2*5)` will reduce the qualified 1,510-parameter design to 1,509 parameters and retain at least 99% accuracy.

INTENDED_EDIT: Reproduce the qualified all-row QKV and positional scalar gauges, then constrain position zero to the single retained Helmert contrast and train positional quotients with dense-coordinate AdamW moments.

EVIDENCE: The 1,510-parameter two-contrast design achieved 99.86%. Removing the other retained contrast failed at 96.98%, so testing this complementary one-parameter quotient is the most direct informative alternative.

<<<<<<< SEARCH
        # Retain the verified balanced query gauges, gauge every key row and
        # every first-head value row, then gauge the first two value rows of
        # the second head. Row 21 uses the ordinary reduced-coordinate update.
        self.gauged_rows = (
            0,
            1,
            head_dim,
            head_dim + 1,
            d_model,
            d_model + 1,
            d_model + 2,
            d_model + 3,
            d_model + head_dim,
            d_model + head_dim + 1,
            d_model + head_dim + 2,
            d_model + head_dim + 3,
            2 * d_model,
            2 * d_model + 1,
            2 * d_model + 2,
            2 * d_model + 3,
            2 * d_model + head_dim,
            2 * d_model + head_dim + 1,
        )
        self.ungauged_rows = tuple(
            row for row in range(self.out_features) if row not in self.gauged_rows
        )

        # Consume the same constructor RNG stream as the replaced nn.Linear.
        base = nn.Linear(d_model, 3 * d_model)
        retained = self.out_features * d_model - len(self.gauged_rows)
        self.weight = nn.Parameter(base.weight.new_empty(retained))
        self.fixed_query_biases = (2, head_dim + 2)
        self.bias = nn.Parameter(
            base.bias.new_empty(d_model - len(self.fixed_query_biases))
        )
=======
        # Gauge every normalized-input row. Query row 6 uses an orthonormal
        # chart; sensitive rows 15, 20, and 23 retain dense AdamW moments.
        self.anchored_rows = (
            0,
            1,
            2,
            3,
            head_dim,
            head_dim + 1,
            head_dim + 3,
            d_model,
            d_model + 1,
            d_model + 2,
            d_model + 3,
            d_model + head_dim,
            d_model + head_dim + 1,
            d_model + head_dim + 2,
            d_model + head_dim + 3,
            2 * d_model,
            2 * d_model + 1,
            2 * d_model + 2,
            2 * d_model + 3,
            2 * d_model + head_dim,
            2 * d_model + head_dim + 1,
            2 * d_model + head_dim + 2,
            2 * d_model + head_dim + 3,
        )
        self.orthonormal_rows = (head_dim + 2,)
        self.gauged_rows = self.anchored_rows + self.orthonormal_rows
        self.ungauged_rows = tuple(
            row for row in range(self.out_features) if row not in self.gauged_rows
        )

        # Consume the same constructor RNG stream as the replaced nn.Linear.
        base = nn.Linear(d_model, 3 * d_model)
        retained = self.out_features * d_model - len(self.gauged_rows)
        self.weight = nn.Parameter(base.weight.new_empty(retained))
        self.fixed_query_biases = (2, head_dim + 2)
        self.bias = nn.Parameter(
            base.bias.new_empty(d_model - len(self.fixed_query_biases))
        )

        basis = torch.zeros(d_model, d_model - 1)
        for column in range(d_model - 1):
            scale = math.sqrt((column + 1) * (column + 2))
            basis[: column + 1, column] = 1.0 / scale
            basis[column + 1, column] = -(column + 1) / scale
        self.register_buffer("input_basis", basis, persistent=False)
>>>>>>> REPLACE

<<<<<<< SEARCH
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        row_width = self.in_features - 1
        split = len(self.gauged_rows) * row_width
        gauged = torch.cat(
            (
                self.weight[:split].view(len(self.gauged_rows), row_width),
                self.weight.new_zeros(len(self.gauged_rows), 1),
            ),
            dim=1,
        )
        ungauged = self.weight[split:].view(
            len(self.ungauged_rows), self.in_features
        )

        rows = []
        gauged_index = 0
        ungauged_index = 0
        for row in range(self.out_features):
            if row in self.gauged_rows:
                rows.append(gauged[gauged_index])
                gauged_index += 1
            else:
                rows.append(ungauged[ungauged_index])
                ungauged_index += 1
        weight = torch.stack(rows)

        query_bias_parts = []
=======
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        row_width = self.in_features - 1
        anchored_split = len(self.anchored_rows) * row_width
        gauged_split = len(self.gauged_rows) * row_width
        anchored = torch.cat(
            (
                self.weight[:anchored_split].view(
                    len(self.anchored_rows), row_width
                ),
                self.weight.new_zeros(len(self.anchored_rows), 1),
            ),
            dim=1,
        )
        orthonormal_reduced = self.weight[
            anchored_split:gauged_split
        ].view(len(self.orthonormal_rows), row_width)
        orthonormal = (
            orthonormal_reduced @ self.input_basis.transpose(0, 1)
        )
        ungauged = self.weight[gauged_split:].view(
            len(self.ungauged_rows), self.in_features
        )

        rows = []
        anchored_index = 0
        orthonormal_index = 0
        ungauged_index = 0
        for row in range(self.out_features):
            if row in self.anchored_rows:
                rows.append(anchored[anchored_index])
                anchored_index += 1
            elif row in self.orthonormal_rows:
                rows.append(orthonormal[orthonormal_index])
                orthonormal_index += 1
            else:
                rows.append(ungauged[ungauged_index])
                ungauged_index += 1
        weight = torch.stack(rows)

        query_bias_parts = []
>>>>>>> REPLACE

<<<<<<< SEARCH
class ThreeCoordinateGaugedPositionEmbedding(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim

        # Consume the same constructor RNG stream as nn.Embedding.
        base = nn.Embedding(num_embeddings, embedding_dim)
        retained = num_embeddings * embedding_dim - 3
        self.weight = nn.Parameter(base.weight.new_empty(retained))
        self.register_buffer(
            "_init_token_shift", base.weight.new_zeros(3), persistent=False
        )

    def forward(self, indices: torch.Tensor) -> torch.Tensor:
        flat = torch.cat(
            (
                self.weight.new_zeros(1),
                self.weight[:2],
                self.weight.new_zeros(1),
                self.weight[2:5],
                self.weight.new_zeros(1),
                self.weight[5:],
            )
        )
        return F.embedding(
            indices, flat.view(self.num_embeddings, self.embedding_dim)
        )
=======
class SingleContrastRowScalarGaugedPositionEmbedding(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.position_zero_coordinates = (2, 4, 5, 6)
        self.position_zero_width = len(self.position_zero_coordinates) - 3

        # Retain only the balanced (2 + 4 - 2*5) contrast at position zero.
        # Every later position is represented in the zero-mean feature space.
        base = nn.Embedding(num_embeddings, embedding_dim)
        retained = self.position_zero_width + (
            num_embeddings - 1
        ) * (embedding_dim - 1)
        self.weight = nn.Parameter(base.weight.new_empty(retained))
        self.register_buffer(
            "_init_token_shift",
            base.weight.new_zeros(embedding_dim),
            persistent=False,
        )

        position_zero_basis = torch.zeros(
            len(self.position_zero_coordinates),
            self.position_zero_width,
        )
        scale = math.sqrt(6.0)
        position_zero_basis[:2, 0] = 1.0 / scale
        position_zero_basis[2, 0] = -2.0 / scale
        self.register_buffer(
            "position_zero_basis", position_zero_basis, persistent=False
        )

        feature_basis = torch.zeros(embedding_dim, embedding_dim - 1)
        for column in range(embedding_dim - 1):
            scale = math.sqrt((column + 1) * (column + 2))
            feature_basis[: column + 1, column] = 1.0 / scale
            feature_basis[column + 1, column] = -(column + 1) / scale
        self.register_buffer("feature_basis", feature_basis, persistent=False)

    def forward(self, indices: torch.Tensor) -> torch.Tensor:
        reduced_zero = self.weight[: self.position_zero_width]
        remaining_zero = self.position_zero_basis @ reduced_zero
        zero = self.weight.new_zeros(())
        position_zero = torch.stack(
            (
                zero,
                zero,
                remaining_zero[0],
                zero,
                remaining_zero[1],
                remaining_zero[2],
                remaining_zero[3],
                zero,
            )
        )

        reduced_rows = self.weight[self.position_zero_width :].view(
            self.num_embeddings - 1, self.embedding_dim - 1
        )
        remaining_positions = reduced_rows @ self.feature_basis.transpose(0, 1)
        dense = torch.cat(
            (position_zero.unsqueeze(0), remaining_positions), dim=0
        )
        return F.embedding(indices, dense)
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.pos_emb = ThreeCoordinateGaugedPositionEmbedding(
            cfg.max_seq_len, cfg.d_model
        )
=======
        self.pos_emb = SingleContrastRowScalarGaugedPositionEmbedding(
            cfg.max_seq_len, cfg.d_model
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.token_emb.transfer_coordinate_shifts(
            (0, 3, 7), self.pos_emb._init_token_shift
        )
=======
        self.token_emb.transfer_coordinate_shifts(
            tuple(range(cfg.d_model)), self.pos_emb._init_token_shift
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
        elif isinstance(module, ThreeCoordinateGaugedPositionEmbedding):
            with torch.no_grad():
                full = module.weight.new_empty(
                    module.num_embeddings, module.embedding_dim
                )
                nn.init.normal_(full, mean=0.0, std=0.02)
                shifts = torch.stack(
                    (full[0, 0], full[0, 3], full[0, 7])
                ).clone()
                full[:, 0].sub_(shifts[0])
                full[:, 3].sub_(shifts[1])
                full[:, 7].sub_(shifts[2])
                flat = full.flatten()
                module.weight.copy_(
                    torch.cat((flat[1:3], flat[4:7], flat[8:]))
                )
                module._init_token_shift.copy_(shifts)
=======
        elif isinstance(
            module, SingleContrastRowScalarGaugedPositionEmbedding
        ):
            with torch.no_grad():
                full = module.weight.new_empty(
                    module.num_embeddings, module.embedding_dim
                )
                nn.init.normal_(full, mean=0.0, std=0.02)

                anchor_coordinates = (0, 1, 3, 7)
                anchor_shifts = full[
                    0, list(anchor_coordinates)
                ].clone()
                for coordinate, shift in zip(
                    anchor_coordinates, anchor_shifts
                ):
                    full[:, coordinate].sub_(shift)

                remaining_values = full[
                    0, list(module.position_zero_coordinates)
                ].clone()
                retained_values = module.position_zero_basis @ (
                    module.position_zero_basis.transpose(0, 1)
                    @ remaining_values
                )
                remaining_shifts = remaining_values - retained_values
                for coordinate, shift in zip(
                    module.position_zero_coordinates, remaining_shifts
                ):
                    full[:, coordinate].sub_(shift)

                token_shifts = torch.stack(
                    (
                        anchor_shifts[0],
                        anchor_shifts[1],
                        remaining_shifts[0],
                        anchor_shifts[2],
                        remaining_shifts[1],
                        remaining_shifts[2],
                        remaining_shifts[3],
                        anchor_shifts[3],
                    )
                )

                full[1:].sub_(full[1:].mean(dim=1, keepdim=True))
                position_zero = (
                    full[0, list(module.position_zero_coordinates)]
                    @ module.position_zero_basis
                )
                reduced_rows = full[1:] @ module.feature_basis
                module.weight.copy_(
                    torch.cat((position_zero, reduced_rows.flatten()))
                )
                module._init_token_shift.copy_(token_shifts)
>>>>>>> REPLACE

<<<<<<< SEARCH
                # LayerNorm scales initialize to one, so subtracting each
                # omitted coefficient preserves every selected row function.
                gauged = full[list(module.gauged_rows)].clone()
                gauged[:, :-1].sub_(gauged[:, -1:])
                ungauged = full[list(module.ungauged_rows)]
                module.weight.copy_(
                    torch.cat((gauged[:, :-1].flatten(), ungauged.flatten()))
                )
                nn.init.zeros_(module.bias)
=======
                # Anchored rows use final-zero representatives, while query
                # row 6 uses orthonormal coordinates for the same quotient.
                anchored = full[list(module.anchored_rows)].clone()
                anchored[:, :-1].sub_(anchored[:, -1:])
                orthonormal = (
                    full[list(module.orthonormal_rows)] @ module.input_basis
                )
                ungauged = full[list(module.ungauged_rows)]
                module.weight.copy_(
                    torch.cat(
                        (
                            anchored[:, :-1].flatten(),
                            orthonormal.flatten(),
                            ungauged.flatten(),
                        )
                    )
                )
                nn.init.zeros_(module.bias)
>>>>>>> REPLACE

<<<<<<< SEARCH
        for dense_row, (start, stop) in zip(
            dense_weight, target_slices
        ):
            parameter[start:stop].copy_(dense_row[:-1])


def save_json(path: Path, obj: Dict) -> None:
=======
        for dense_row, (start, stop) in zip(
            dense_weight, target_slices
        ):
            parameter[start:stop].copy_(dense_row[:-1])


@torch.no_grad()
def step_dense_position_row_quotients(
    specifications, states, lr: float, weight_decay: float
) -> None:
    """Apply dense AdamW to all positional residual-scalar quotients."""
    beta1, beta2, eps = 0.9, 0.999, 1e-8

    for module, state in zip(specifications, states):
        parameter = module.weight
        if parameter.grad is None:
            continue

        start = module.position_zero_width
        anchored_zero = parameter[:start].clone()
        zero_reduced_grad = parameter.grad[:start]
        zero_dense_grad = (
            module.position_zero_basis @ zero_reduced_grad
        )

        anchored_rows = parameter[start:].view(
            module.num_embeddings - 1, module.embedding_dim - 1
        ).clone()
        row_reduced_grad = parameter.grad[start:].view_as(anchored_rows)
        row_dense_grad = (
            row_reduced_grad @ module.feature_basis.transpose(0, 1)
        )

        state["step"] += 1
        step = state["step"]
        bias_correction1 = 1.0 - beta1 ** step
        bias_correction2 = 1.0 - beta2 ** step

        zero_exp_avg = state["zero_exp_avg"]
        zero_exp_avg_sq = state["zero_exp_avg_sq"]
        zero_exp_avg.mul_(beta1).add_(
            zero_dense_grad, alpha=1.0 - beta1
        )
        zero_exp_avg_sq.mul_(beta2).addcmul_(
            zero_dense_grad, zero_dense_grad, value=1.0 - beta2
        )
        zero_denom = zero_exp_avg_sq.sqrt().div_(
            math.sqrt(bias_correction2)
        ).add_(eps)
        dense_zero = module.position_zero_basis @ anchored_zero
        dense_zero.mul_(1.0 - lr * weight_decay)
        dense_zero.add_(
            zero_exp_avg / zero_denom,
            alpha=-lr / bias_correction1,
        )
        parameter[:start].copy_(
            module.position_zero_basis.transpose(0, 1) @ dense_zero
        )

        dense_exp_avg = state["dense_exp_avg"]
        dense_exp_avg_sq = state["dense_exp_avg_sq"]
        dense_exp_avg.mul_(beta1).add_(
            row_dense_grad, alpha=1.0 - beta1
        )
        dense_exp_avg_sq.mul_(beta2).addcmul_(
            row_dense_grad, row_dense_grad, value=1.0 - beta2
        )
        dense_denom = dense_exp_avg_sq.sqrt().div_(
            math.sqrt(bias_correction2)
        ).add_(eps)
        dense_rows = anchored_rows @ module.feature_basis.transpose(0, 1)
        dense_rows.mul_(1.0 - lr * weight_decay)
        dense_rows.add_(
            dense_exp_avg / dense_denom,
            alpha=-lr / bias_correction1,
        )
        parameter[start:].copy_(
            (dense_rows @ module.feature_basis).flatten()
        )


def save_json(path: Path, obj: Dict) -> None:
>>>>>>> REPLACE

<<<<<<< SEARCH
    # Residual-output quotients retain dense output-coordinate moments. The
    # completed final key row and new second-head value row additionally use
    # dense moments for their omitted normalized-input coefficients.
    quotient_specifications = []
    qkv_row_specifications = []
    for block in model.blocks:
        quotient_specifications.extend(
            (
                (block.attn.proj, "weight"),
                (block.mlp.fc2, "weight"),
                (block.mlp.fc2, "bias"),
            )
        )
        qkv = block.attn.qkv
        qkv_row_specifications.append(
            (
                qkv,
                block.ln1,
                (
                    2 * qkv.in_features - 1,
                    2 * qkv.in_features + block.attn.head_dim,
                ),
            )
        )
    quotient_parameter_ids = {
        id(getattr(module, parameter_name))
        for module, parameter_name in quotient_specifications
    }
    qkv_row_parameter_ids = {
        id(module.weight)
        for module, _, _ in qkv_row_specifications
    }
    optimizer = torch.optim.AdamW(
        [
            parameter
            for parameter in model.parameters()
            if id(parameter) not in quotient_parameter_ids
            and id(parameter) not in qkv_row_parameter_ids
        ],
        lr=train_cfg.lr,
        weight_decay=train_cfg.weight_decay,
    )
    qkv_row_states = []
    for module, _, target_rows in qkv_row_specifications:
        parameter = module.weight
        qkv_row_states.append(
            {
                "step": 0,
                "exp_avg": parameter.new_zeros(parameter.shape),
                "exp_avg_sq": parameter.new_zeros(parameter.shape),
                "dense_exp_avg": parameter.new_zeros(
                    len(target_rows), module.in_features
                ),
                "dense_exp_avg_sq": parameter.new_zeros(
                    len(target_rows), module.in_features
                ),
            }
        )
    quotient_states = []
=======
    # Residual-output, sensitive QKV, and positional quotients retain AdamW
    # moments in their original dense coordinate systems.
    quotient_specifications = []
    qkv_row_specifications = []
    for block in model.blocks:
        quotient_specifications.extend(
            (
                (block.attn.proj, "weight"),
                (block.mlp.fc2, "weight"),
                (block.mlp.fc2, "bias"),
            )
        )
        qkv = block.attn.qkv
        qkv_row_specifications.append(
            (
                qkv,
                block.ln1,
                (
                    2 * qkv.in_features - 1,
                    2 * qkv.in_features + block.attn.head_dim,
                    3 * qkv.in_features - 1,
                ),
            )
        )
    quotient_parameter_ids = {
        id(getattr(module, parameter_name))
        for module, parameter_name in quotient_specifications
    }
    qkv_row_parameter_ids = {
        id(module.weight)
        for module, _, _ in qkv_row_specifications
    }
    position_row_specifications = [model.pos_emb]
    position_row_parameter_ids = {
        id(module.weight) for module in position_row_specifications
    }
    optimizer = torch.optim.AdamW(
        [
            parameter
            for parameter in model.parameters()
            if id(parameter) not in quotient_parameter_ids
            and id(parameter) not in qkv_row_parameter_ids
            and id(parameter) not in position_row_parameter_ids
        ],
        lr=train_cfg.lr,
        weight_decay=train_cfg.weight_decay,
    )
    qkv_row_states = []
    for module, _, target_rows in qkv_row_specifications:
        parameter = module.weight
        qkv_row_states.append(
            {
                "step": 0,
                "exp_avg": parameter.new_zeros(parameter.shape),
                "exp_avg_sq": parameter.new_zeros(parameter.shape),
                "dense_exp_avg": parameter.new_zeros(
                    len(target_rows), module.in_features
                ),
                "dense_exp_avg_sq": parameter.new_zeros(
                    len(target_rows), module.in_features
                ),
            }
        )
    position_row_states = []
    for module in position_row_specifications:
        parameter = module.weight
        position_row_states.append(
            {
                "step": 0,
                "zero_exp_avg": parameter.new_zeros(
                    len(module.position_zero_coordinates)
                ),
                "zero_exp_avg_sq": parameter.new_zeros(
                    len(module.position_zero_coordinates)
                ),
                "dense_exp_avg": parameter.new_zeros(
                    module.num_embeddings - 1, module.embedding_dim
                ),
                "dense_exp_avg_sq": parameter.new_zeros(
                    module.num_embeddings - 1, module.embedding_dim
                ),
            }
        )
    quotient_states = []
>>>>>>> REPLACE

<<<<<<< SEARCH
        step_dense_qkv_row_quotients(
            qkv_row_specifications,
            qkv_row_states,
            lr_now,
            train_cfg.weight_decay,
        )
        optimizer.step()
=======
        step_dense_qkv_row_quotients(
            qkv_row_specifications,
            qkv_row_states,
            lr_now,
            train_cfg.weight_decay,
        )
        step_dense_position_row_quotients(
            position_row_specifications,
            position_row_states,
            lr_now,
            train_cfg.weight_decay,
        )
        optimizer.step()
>>>>>>> REPLACE