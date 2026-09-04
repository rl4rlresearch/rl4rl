MECHANISM: Validation-efficient orthonormal tied-token translation quotient

HYPOTHESIS: Reproducing the qualified 1,509-parameter single-contrast design, then quotienting its final token–position translation with an orthonormal token-zero chart will yield 1,508 parameters and at least 99% accuracy; reducing validation frequency will avoid the timeouts seen in the prior 1,508-parameter attempts.

INTENDED_EDIT: Retain only the successful `(2 + 4 - 2*5)` position-zero contrast, constrain token zero orthogonally to that translation direction, transfer omitted initialization components across every positional row, preserve dense-coordinate positional AdamW, and evaluate every 600 steps.

EVIDENCE: The single-contrast design achieved 99.98% at 1,509 parameters. Both subsequent 1,508-parameter tied-token quotient attempts timed out rather than failing accuracy, so this patch preserves the orthonormal quotient while reducing validation overhead.

<<<<<<< SEARCH
class ScalarGaugedTokenEmbedding(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim

        # Consume the same constructor RNG stream as nn.Embedding.
        base = nn.Embedding(num_embeddings, embedding_dim)
        retained = num_embeddings * embedding_dim - 1
        self.weight = nn.Parameter(base.weight.new_empty(retained))

    def dense_weight(self) -> torch.Tensor:
        flat = torch.cat((self.weight.new_zeros(1), self.weight))
        return flat.view(self.num_embeddings, self.embedding_dim)

    def forward(self, indices: torch.Tensor) -> torch.Tensor:
        return F.embedding(indices, self.dense_weight())

    @torch.no_grad()
    def transfer_coordinate_shifts(self, coordinates, shifts) -> None:
        full = self.dense_weight().clone()
        for coordinate, shift in zip(coordinates, shifts):
            full[:, coordinate].add_(shift)

        # A common scalar translation of every embedding coordinate is
        # invisible to all LayerNorms and adds only a common output-logit shift.
        anchor = full[0, 0].clone()
        full.sub_(anchor)
        self.weight.copy_(full.flatten()[1:])
=======
class ScalarGaugedTokenEmbedding(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.token_zero_width = embedding_dim - 2

        # Consume the same constructor RNG stream as nn.Embedding. Token zero
        # retains the scalar anchor and is orthogonal to the final learned
        # token-position translation direction.
        base = nn.Embedding(num_embeddings, embedding_dim)
        retained = self.token_zero_width + (
            num_embeddings - 1
        ) * embedding_dim
        self.weight = nn.Parameter(base.weight.new_empty(retained))
        self.register_buffer(
            "_init_position_shift", base.weight.new_zeros(()), persistent=False
        )

        translation_direction = torch.zeros(embedding_dim)
        scale = math.sqrt(6.0)
        translation_direction[2] = 1.0 / scale
        translation_direction[4] = 1.0 / scale
        translation_direction[5] = -2.0 / scale
        self.register_buffer(
            "translation_direction",
            translation_direction,
            persistent=False,
        )

        complete_basis, _ = torch.linalg.qr(
            translation_direction[1:].unsqueeze(1), mode="complete"
        )
        self.register_buffer(
            "token_zero_basis",
            complete_basis[:, 1:],
            persistent=False,
        )

    def dense_weight(self) -> torch.Tensor:
        reduced_zero = self.weight[: self.token_zero_width]
        token_zero = torch.cat(
            (
                self.weight.new_zeros(1),
                self.token_zero_basis @ reduced_zero,
            )
        )
        remaining = self.weight[self.token_zero_width :].view(
            self.num_embeddings - 1, self.embedding_dim
        )
        return torch.cat((token_zero.unsqueeze(0), remaining), dim=0)

    def forward(self, indices: torch.Tensor) -> torch.Tensor:
        return F.embedding(indices, self.dense_weight())

    @torch.no_grad()
    def transfer_coordinate_shifts(
        self, coordinates, shifts, position_embedding
    ) -> None:
        full = self.dense_weight().clone()
        for coordinate, shift in zip(coordinates, shifts):
            full[:, coordinate].add_(shift)

        # Restore the scalar representative, then transfer the final omitted
        # contrast from every token row into every positional row.
        anchor = full[0, 0].clone()
        full.sub_(anchor)
        position_shift = full[0] @ self.translation_direction
        full.sub_(position_shift * self.translation_direction)
        self.weight.copy_(
            torch.cat(
                (
                    full[0, 1:] @ self.token_zero_basis,
                    full[1:].flatten(),
                )
            )
        )
        position_embedding.add_translation(
            self._init_position_shift + position_shift
        )
>>>>>>> REPLACE

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
class SingleContrastRowScalarGaugedPositionEmbedding(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.position_zero_coordinates = (2, 4, 5, 6)
        self.position_zero_width = 1

        # Retain only the qualified (2 + 4 - 2*5) position-zero contrast.
        # Every later position remains in the zero-mean feature subspace.
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

        translation_direction = torch.zeros(embedding_dim)
        scale = math.sqrt(6.0)
        translation_direction[2] = 1.0 / scale
        translation_direction[4] = 1.0 / scale
        translation_direction[5] = -2.0 / scale
        self.register_buffer(
            "translation_direction",
            translation_direction,
            persistent=False,
        )
        self.register_buffer(
            "position_zero_basis",
            translation_direction[
                list(self.position_zero_coordinates)
            ].unsqueeze(1),
            persistent=False,
        )

        feature_basis = torch.zeros(embedding_dim, embedding_dim - 1)
        for column in range(embedding_dim - 1):
            scale = math.sqrt((column + 1) * (column + 2))
            feature_basis[: column + 1, column] = 1.0 / scale
            feature_basis[column + 1, column] = -(column + 1) / scale
        self.register_buffer("feature_basis", feature_basis, persistent=False)

    def forward(self, indices: torch.Tensor) -> torch.Tensor:
        remaining_zero = (
            self.position_zero_basis
            @ self.weight[: self.position_zero_width]
        )
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

    @torch.no_grad()
    def add_translation(self, coefficient) -> None:
        self.weight[0].add_(coefficient)
        reduced_direction = self.translation_direction @ self.feature_basis
        self.weight[self.position_zero_width :].view(
            self.num_embeddings - 1, self.embedding_dim - 1
        ).add_(coefficient * reduced_direction)
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.pos_emb = FourCoordinateRowScalarGaugedPositionEmbedding(
            cfg.max_seq_len, cfg.d_model
        )
=======
        self.pos_emb = SingleContrastRowScalarGaugedPositionEmbedding(
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
        # Transfer all omitted positional directions, then restore both tied
        # token gauges while preserving the retained positional contrast.
        self.token_emb.transfer_coordinate_shifts(
            tuple(range(cfg.d_model)),
            self.pos_emb._init_token_shift,
            self.pos_emb,
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
        if isinstance(module, ScalarGaugedTokenEmbedding):
            with torch.no_grad():
                full = module.weight.new_empty(
                    module.num_embeddings, module.embedding_dim
                )
                nn.init.normal_(full, mean=0.0, std=0.02)
                anchor = full[0, 0].clone()
                full.sub_(anchor)
                module.weight.copy_(full.flatten()[1:])
        elif isinstance(module, TiedGaugedLMHead):
            with torch.no_grad():
                embedding = module._embedding
                full = embedding.weight.new_empty(
                    embedding.num_embeddings, embedding.embedding_dim
                )
                nn.init.normal_(full, mean=0.0, std=0.02)
                anchor = full[0, 0].clone()
                full.sub_(anchor)
                embedding.weight.copy_(full.flatten()[1:])
=======
        if isinstance(module, ScalarGaugedTokenEmbedding):
            with torch.no_grad():
                full = module.weight.new_empty(
                    module.num_embeddings, module.embedding_dim
                )
                nn.init.normal_(full, mean=0.0, std=0.02)
                anchor = full[0, 0].clone()
                full.sub_(anchor)
                position_shift = full[0] @ module.translation_direction
                full.sub_(position_shift * module.translation_direction)
                module.weight.copy_(
                    torch.cat(
                        (
                            full[0, 1:] @ module.token_zero_basis,
                            full[1:].flatten(),
                        )
                    )
                )
                module._init_position_shift.copy_(position_shift)
        elif isinstance(module, TiedGaugedLMHead):
            with torch.no_grad():
                embedding = module._embedding
                full = embedding.weight.new_empty(
                    embedding.num_embeddings, embedding.embedding_dim
                )
                nn.init.normal_(full, mean=0.0, std=0.02)
                anchor = full[0, 0].clone()
                full.sub_(anchor)
                position_shift = full[0] @ embedding.translation_direction
                full.sub_(position_shift * embedding.translation_direction)
                embedding.weight.copy_(
                    torch.cat(
                        (
                            full[0, 1:] @ embedding.token_zero_basis,
                            full[1:].flatten(),
                        )
                    )
                )
                embedding._init_position_shift.copy_(position_shift)
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

<<<<<<< SEARCH
    p.add_argument("--eval-interval", type=int, default=300)
=======
    p.add_argument("--eval-interval", type=int, default=600)
>>>>>>> REPLACE