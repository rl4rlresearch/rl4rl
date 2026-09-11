MECHANISM: Orthonormal final-query-row LayerNorm quotient

HYPOTHESIS: Adding second-head query row 6 through a Helmert orthonormal chart to the verified 1,513-parameter position-scalar design will produce a 1,512-parameter model with at least 99% accuracy.

INTENDED_EDIT: Reproduce the qualified dense-AdamW position-row scalar quotient, then remove query row 6’s exact normalized-input null direction using an orthonormal seven-coordinate basis while retaining the verified dense updates for rows 15, 20, and 23.

EVIDENCE: The position-scalar quotient achieved 99.95% at 1,513 parameters, while query row 6 failed in a last-coordinate reduced chart at 1,512 parameters. Query row 7 succeeded under the same capacity reduction, so testing a symmetric orthonormal chart directly targets optimizer geometry without removing additional function capacity.

<<<<<<< SEARCH
        # Retain the verified query-row-7 design and every key and value
        # gauge. Final rows 15 and 23 and value row 20 use recovered
        # dense-coordinate AdamW moments.
        self.gauged_rows = (
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
        # Gauge every query, key, and value row. The previously sensitive
        # second-head query row 6 uses an orthonormal zero-mean chart; rows 15,
        # 20, and 23 retain recovered dense-coordinate AdamW moments.
        self.orthogonal_gauged_rows = (head_dim + 2,)
        self.gauged_rows = (
            0,
            1,
            2,
            3,
            head_dim,
            head_dim + 1,
            head_dim + 2,
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
=======
        row_width = self.in_features - 1
        split = len(self.gauged_rows) * row_width
        reduced_rows = self.weight[:split].view(
            len(self.gauged_rows), row_width
        )
        gauged_rows = []
        for row, reduced_row in zip(self.gauged_rows, reduced_rows):
            if row in self.orthogonal_gauged_rows:
                gauged_rows.append(self.input_basis @ reduced_row)
            else:
                gauged_rows.append(
                    torch.cat((reduced_row, reduced_row.new_zeros(1)))
                )
        gauged = torch.stack(gauged_rows)
        ungauged = self.weight[split:].view(
>>>>>>> REPLACE

<<<<<<< SEARCH
class FourCoordinateGaugedPositionEmbedding(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim

        # Consume the same constructor RNG stream as nn.Embedding.
        base = nn.Embedding(num_embeddings, embedding_dim)
        retained = num_embeddings * embedding_dim - 4
        self.weight = nn.Parameter(base.weight.new_empty(retained))
        self.register_buffer(
            "_init_token_shift", base.weight.new_zeros(4), persistent=False
        )

    def forward(self, indices: torch.Tensor) -> torch.Tensor:
        flat = torch.cat(
            (
                self.weight.new_zeros(2),
                self.weight[:1],
                self.weight.new_zeros(1),
                self.weight[1:4],
                self.weight.new_zeros(1),
                self.weight[4:],
            )
        )
        return F.embedding(
            indices, flat.view(self.num_embeddings, self.embedding_dim)
        )
=======
class FourCoordinateRowScalarGaugedPositionEmbedding(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.position_zero_width = embedding_dim - 4

        # Position zero fixes four token-position translations. Every later
        # position is represented modulo its residual-stream scalar shift.
        base = nn.Embedding(num_embeddings, embedding_dim)
        retained = self.position_zero_width + (
            num_embeddings - 1
        ) * (embedding_dim - 1)
        self.weight = nn.Parameter(base.weight.new_empty(retained))
        self.register_buffer(
            "_init_token_shift", base.weight.new_zeros(4), persistent=False
        )

        basis = torch.zeros(embedding_dim, embedding_dim - 1)
        for column in range(embedding_dim - 1):
            scale = math.sqrt((column + 1) * (column + 2))
            basis[: column + 1, column] = 1.0 / scale
            basis[column + 1, column] = -(column + 1) / scale
        self.register_buffer("feature_basis", basis, persistent=False)

    def forward(self, indices: torch.Tensor) -> torch.Tensor:
        position_zero = torch.cat(
            (
                self.weight.new_zeros(2),
                self.weight[:1],
                self.weight.new_zeros(1),
                self.weight[1:4],
                self.weight.new_zeros(1),
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
        self.pos_emb = FourCoordinateGaugedPositionEmbedding(
=======
        self.pos_emb = FourCoordinateRowScalarGaugedPositionEmbedding(
>>>>>>> REPLACE

<<<<<<< SEARCH
        elif isinstance(module, FourCoordinateGaugedPositionEmbedding):
            with torch.no_grad():
                full = module.weight.new_empty(
                    module.num_embeddings, module.embedding_dim
                )
                nn.init.normal_(full, mean=0.0, std=0.02)
                shifts = torch.stack(
                    (full[0, 0], full[0, 1], full[0, 3], full[0, 7])
                ).clone()
                full[:, 0].sub_(shifts[0])
                full[:, 1].sub_(shifts[1])
                full[:, 3].sub_(shifts[2])
                full[:, 7].sub_(shifts[3])
                flat = full.flatten()
                module.weight.copy_(
                    torch.cat((flat[2:3], flat[4:7], flat[8:]))
                )
                module._init_token_shift.copy_(shifts)
=======
        elif isinstance(module, FourCoordinateRowScalarGaugedPositionEmbedding):
            with torch.no_grad():
                full = module.weight.new_empty(
                    module.num_embeddings, module.embedding_dim
                )
                nn.init.normal_(full, mean=0.0, std=0.02)
                shifts = torch.stack(
                    (full[0, 0], full[0, 1], full[0, 3], full[0, 7])
                ).clone()
                full[:, 0].sub_(shifts[0])
                full[:, 1].sub_(shifts[1])
                full[:, 3].sub_(shifts[2])
                full[:, 7].sub_(shifts[3])

                full[1:].sub_(full[1:].mean(dim=1, keepdim=True))
                position_zero = torch.cat(
                    (full[0, 2:3], full[0, 4:7])
                )
                reduced_rows = full[1:] @ module.feature_basis
                module.weight.copy_(
                    torch.cat((position_zero, reduced_rows.flatten()))
                )
                module._init_token_shift.copy_(shifts)
>>>>>>> REPLACE

<<<<<<< SEARCH
                # Attention scales initialize to one, so subtracting each
                # omitted coefficient preserves both selected row functions.
                gauged = full[list(module.gauged_rows)].clone()
                gauged[:, :-1].sub_(gauged[:, -1:])
                ungauged = full[list(module.ungauged_rows)]
                module.weight.copy_(
                    torch.cat((gauged[:, :-1].flatten(), ungauged.flatten()))
                )
=======
                # At initialization the attention scales are one. Use a
                # symmetric Helmert representative for row 6 and the verified
                # zero-final-coordinate representatives for all other rows.
                gauged_reduced = []
                for row in module.gauged_rows:
                    dense_row = full[row]
                    if row in module.orthogonal_gauged_rows:
                        gauged_reduced.append(
                            module.input_basis.transpose(0, 1) @ dense_row
                        )
                    else:
                        gauged_reduced.append(
                            dense_row[:-1] - dense_row[-1]
                        )
                ungauged = full[list(module.ungauged_rows)]
                module.weight.copy_(
                    torch.cat(
                        (
                            torch.stack(gauged_reduced).flatten(),
                            ungauged.flatten(),
                        )
                    )
                )
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
    """Apply dense AdamW to position rows modulo scalar residual shifts."""
    beta1, beta2, eps = 0.9, 0.999, 1e-8

    for module, state in zip(specifications, states):
        parameter = module.weight
        if parameter.grad is None:
            continue

        start = module.position_zero_width
        anchored_reduced = parameter[start:].view(
            module.num_embeddings - 1, module.embedding_dim - 1
        ).clone()
        reduced_grad = parameter.grad[start:].view_as(anchored_reduced)
        dense_grad = reduced_grad @ module.feature_basis.transpose(0, 1)

        state["step"] += 1
        step = state["step"]
        bias_correction1 = 1.0 - beta1 ** step
        bias_correction2 = 1.0 - beta2 ** step

        # Preserve ordinary reduced-coordinate AdamW for position zero.
        exp_avg = state["exp_avg"]
        exp_avg_sq = state["exp_avg_sq"]
        exp_avg.mul_(beta1).add_(parameter.grad, alpha=1.0 - beta1)
        exp_avg_sq.mul_(beta2).addcmul_(
            parameter.grad, parameter.grad, value=1.0 - beta2
        )
        denom = exp_avg_sq.sqrt().div_(
            math.sqrt(bias_correction2)
        ).add_(eps)
        parameter.mul_(1.0 - lr * weight_decay)
        parameter.add_(
            exp_avg / denom,
            alpha=-lr / bias_correction1,
        )

        # Update later rows in their original dense coordinates, then project
        # away only the functionally invisible scalar component.
        dense_exp_avg = state["dense_exp_avg"]
        dense_exp_avg_sq = state["dense_exp_avg_sq"]
        dense_exp_avg.mul_(beta1).add_(dense_grad, alpha=1.0 - beta1)
        dense_exp_avg_sq.mul_(beta2).addcmul_(
            dense_grad, dense_grad, value=1.0 - beta2
        )
        dense_denom = dense_exp_avg_sq.sqrt().div_(
            math.sqrt(bias_correction2)
        ).add_(eps)
        dense_weight = (
            anchored_reduced @ module.feature_basis.transpose(0, 1)
        )
        dense_weight.mul_(1.0 - lr * weight_decay)
        dense_weight.add_(
            dense_exp_avg / dense_denom,
            alpha=-lr / bias_correction1,
        )
        parameter[start:].copy_(
            (dense_weight @ module.feature_basis).flatten()
        )


def save_json(path: Path, obj: Dict) -> None:
>>>>>>> REPLACE

<<<<<<< SEARCH
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
=======
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
>>>>>>> REPLACE

<<<<<<< SEARCH
    quotient_states = []
    for module, parameter_name in quotient_specifications:
=======
    position_row_states = []
    for module in position_row_specifications:
        parameter = module.weight
        position_row_states.append(
            {
                "step": 0,
                "exp_avg": parameter.new_zeros(parameter.shape),
                "exp_avg_sq": parameter.new_zeros(parameter.shape),
                "dense_exp_avg": parameter.new_zeros(
                    module.num_embeddings - 1, module.embedding_dim
                ),
                "dense_exp_avg_sq": parameter.new_zeros(
                    module.num_embeddings - 1, module.embedding_dim
                ),
            }
        )
    quotient_states = []
    for module, parameter_name in quotient_specifications:
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