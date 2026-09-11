MECHANISM: Third orthonormal token–position translation quotient

HYPOTHESIS: Removing position zero’s remaining Helmert contrast proportional to coordinates `(2 + 4 - 2*5)` will reduce the verified 1,510-parameter design to 1,509 parameters while retaining at least 99% accuracy.

INTENDED_EDIT: Reproduce the verified 1,510-parameter positional chart, retain only its final Helmert coordinate, transfer all three omitted position-zero directions into the tied token embedding at initialization, and optimize the retained coordinate with dense four-coordinate AdamW moments.

EVIDENCE: The 1,510-parameter design achieved 99.86% after removing the common direction and coordinate-2-versus-4 contrast from position zero; its large margin and the prior success of orthonormal charts motivate testing the next balanced Helmert contrast instead of another fragile single-coordinate anchor.

<<<<<<< SEARCH
class FourCoordinateRowScalarGaugedPositionEmbedding(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.position_zero_width = embedding_dim - 4

        # Consume the same constructor RNG stream as nn.Embedding. Position
        # zero fixes four token-position translations. Every later position is
        # represented in the zero-mean feature subspace because a scalar shift
        # at one position is erased by the pre-norm residual network.
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
        # Position zero anchors coordinates 0, 1, 3, and 7.
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
=======
class FourCoordinateThreeDirectionGaugedPositionEmbedding(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.position_zero_coordinates = (2, 4, 5, 6)
        self.position_zero_width = 1

        # Position zero anchors four coordinates and retains only the final
        # Helmert direction of its four remaining coordinates. Every later
        # position remains in the full zero-mean feature subspace.
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
        helmert_column = 2
        scale = math.sqrt(
            (helmert_column + 1) * (helmert_column + 2)
        )
        position_zero_basis[: helmert_column + 1, 0] = 1.0 / scale
        position_zero_basis[helmert_column + 1, 0] = (
            -(helmert_column + 1) / scale
        )
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
        self.pos_emb = FourCoordinateRowScalarGaugedPositionEmbedding(
            cfg.max_seq_len, cfg.d_model
        )
=======
        self.pos_emb = FourCoordinateThreeDirectionGaugedPositionEmbedding(
            cfg.max_seq_len, cfg.d_model
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
        # Transfer the four positional anchors, then restore the token
        # embedding's scalar translation gauge.
        self.token_emb.transfer_coordinate_shifts(
            (0, 1, 3, 7), self.pos_emb._init_token_shift
        )
=======
        # Transfer every omitted position-zero direction, then restore the
        # token embedding's scalar translation gauge.
        self.token_emb.transfer_coordinate_shifts(
            tuple(range(cfg.d_model)), self.pos_emb._init_token_shift
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
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

                # Choose the zero-mean representative of each independent
                # residual-stream scalar equivalence class.
                full[1:].sub_(full[1:].mean(dim=1, keepdim=True))
                position_zero = torch.cat(
                    (full[0, 2:3], full[0, 4:7])
                )
                reduced_rows = full[1:] @ module.feature_basis
                module.weight.copy_(
                    torch.cat((position_zero, reduced_rows.flatten()))
                )
                module._init_token_shift.copy_(shifts)
=======
        elif isinstance(
            module, FourCoordinateThreeDirectionGaugedPositionEmbedding
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

        # Update all later rows in their original dense coordinates and
        # project away only the functionally invisible scalar component.
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
=======
@torch.no_grad()
def step_dense_position_row_quotients(
    specifications, states, lr: float, weight_decay: float
) -> None:
    """Apply dense AdamW to all positional translation quotients."""
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
>>>>>>> REPLACE

<<<<<<< SEARCH
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
=======
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
>>>>>>> REPLACE