MECHANISM: Dense-AdamW fifth token–position translation quotient

HYPOTHESIS: Anchoring position-zero coordinate 2 while recovering dense AdamW moments for its coupled token/position column will produce a 1,534-parameter model with at least 99% accuracy.

INTENDED_EDIT: Extend the verified four-coordinate positional gauge to coordinate 2 and optimize that quotient in its original dense coordinates while leaving all other embedding coordinates on ordinary reduced-coordinate AdamW.

EVIDENCE: The same fifth anchor reached only 98.05% with reduced-coordinate AdamW, while dense-coordinate optimization rescued other sensitive exact quotients, including key row 15 from 90.92% to 99.87%; the verified four-anchor model reached 99.82%.

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
class FiveCoordinateGaugedPositionEmbedding(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.anchored_coordinates = (0, 1, 2, 3, 7)

        # Consume the same constructor RNG stream as nn.Embedding.
        base = nn.Embedding(num_embeddings, embedding_dim)
        retained = num_embeddings * embedding_dim - 5
        self.weight = nn.Parameter(base.weight.new_empty(retained))
        self.register_buffer(
            "_init_token_shift", base.weight.new_zeros(5), persistent=False
        )

    def dense_weight(self) -> torch.Tensor:
        flat = torch.cat(
            (
                self.weight.new_zeros(4),
                self.weight[:3],
                self.weight.new_zeros(1),
                self.weight[3:],
            )
        )
        return flat.view(self.num_embeddings, self.embedding_dim)

    def forward(self, indices: torch.Tensor) -> torch.Tensor:
        return F.embedding(indices, self.dense_weight())
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.pos_emb = FourCoordinateGaugedPositionEmbedding(
            cfg.max_seq_len, cfg.d_model
        )
=======
        self.pos_emb = FiveCoordinateGaugedPositionEmbedding(
            cfg.max_seq_len, cfg.d_model
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.token_emb.transfer_coordinate_shifts(
            (0, 1, 3, 7), self.pos_emb._init_token_shift
        )
=======
        self.token_emb.transfer_coordinate_shifts(
            (0, 1, 2, 3, 7), self.pos_emb._init_token_shift
        )
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
        elif isinstance(module, FiveCoordinateGaugedPositionEmbedding):
            with torch.no_grad():
                full = module.weight.new_empty(
                    module.num_embeddings, module.embedding_dim
                )
                nn.init.normal_(full, mean=0.0, std=0.02)
                shifts = torch.stack(
                    (
                        full[0, 0],
                        full[0, 1],
                        full[0, 2],
                        full[0, 3],
                        full[0, 7],
                    )
                ).clone()
                for coordinate, shift in zip(
                    module.anchored_coordinates, shifts
                ):
                    full[:, coordinate].sub_(shift)
                flat = full.flatten()
                module.weight.copy_(
                    torch.cat((flat[4:7], flat[8:]))
                )
                module._init_token_shift.copy_(shifts)
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
def step_dense_embedding_quotients(
    specifications, states, lr: float, weight_decay: float
) -> None:
    """Use dense AdamW moments for selected token-position translations."""
    beta1, beta2, eps = 0.9, 0.999, 1e-8

    for (token, position, coordinate), state in zip(
        specifications, states
    ):
        token_parameter = token.weight
        position_parameter = position.weight
        if token_parameter.grad is None or position_parameter.grad is None:
            continue

        dense_token_column = token.dense_weight()[:, coordinate].clone()
        dense_position = position.dense_weight()
        dense_position_column = dense_position[:, coordinate].clone()

        dense_token_grad = torch.cat(
            (
                token_parameter.grad.new_zeros(1),
                token_parameter.grad,
            )
        ).view(token.num_embeddings, token.embedding_dim)[:, coordinate]

        free_mask = torch.ones_like(dense_position, dtype=torch.bool)
        free_mask[0, list(position.anchored_coordinates)] = False
        dense_position_grad = dense_position.new_zeros(dense_position.shape)
        dense_position_grad[free_mask] = position_parameter.grad
        dense_position_grad[0, coordinate] = (
            dense_token_grad.sum()
            - dense_position_grad[1:, coordinate].sum()
        )
        dense_position_column_grad = dense_position_grad[:, coordinate]

        state["step"] += 1
        step = state["step"]
        bias_correction1 = 1.0 - beta1 ** step
        bias_correction2 = 1.0 - beta2 ** step

        # Preserve ordinary reduced-coordinate AdamW for every coefficient
        # outside the newly anchored coordinate.
        for parameter, exp_avg, exp_avg_sq in (
            (
                token_parameter,
                state["token_exp_avg"],
                state["token_exp_avg_sq"],
            ),
            (
                position_parameter,
                state["position_exp_avg"],
                state["position_exp_avg_sq"],
            ),
        ):
            exp_avg.mul_(beta1).add_(
                parameter.grad, alpha=1.0 - beta1
            )
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

        token_dense_exp_avg = state["dense_token_exp_avg"]
        token_dense_exp_avg_sq = state["dense_token_exp_avg_sq"]
        token_dense_exp_avg.mul_(beta1).add_(
            dense_token_grad, alpha=1.0 - beta1
        )
        token_dense_exp_avg_sq.mul_(beta2).addcmul_(
            dense_token_grad, dense_token_grad, value=1.0 - beta2
        )
        token_dense_denom = token_dense_exp_avg_sq.sqrt().div_(
            math.sqrt(bias_correction2)
        ).add_(eps)
        dense_token_column.mul_(1.0 - lr * weight_decay)
        dense_token_column.add_(
            token_dense_exp_avg / token_dense_denom,
            alpha=-lr / bias_correction1,
        )

        position_dense_exp_avg = state["dense_position_exp_avg"]
        position_dense_exp_avg_sq = state["dense_position_exp_avg_sq"]
        position_dense_exp_avg.mul_(beta1).add_(
            dense_position_column_grad, alpha=1.0 - beta1
        )
        position_dense_exp_avg_sq.mul_(beta2).addcmul_(
            dense_position_column_grad,
            dense_position_column_grad,
            value=1.0 - beta2,
        )
        position_dense_denom = position_dense_exp_avg_sq.sqrt().div_(
            math.sqrt(bias_correction2)
        ).add_(eps)
        dense_position_column.mul_(1.0 - lr * weight_decay)
        dense_position_column.add_(
            position_dense_exp_avg / position_dense_denom,
            alpha=-lr / bias_correction1,
        )

        # Return to the position-zero anchor while transferring its shift to
        # the tied token column, preserving inputs and output probabilities.
        shift = dense_position_column[0].clone()
        dense_position_column.sub_(shift)
        dense_token_column.add_(shift)

        updated_token = token.dense_weight().clone()
        updated_token[:, coordinate].copy_(dense_token_column)
        token_parameter.copy_(updated_token.flatten()[1:])

        updated_position = position.dense_weight().clone()
        updated_position[:, coordinate].copy_(dense_position_column)
        position_parameter.copy_(updated_position[free_mask])


def save_json(path: Path, obj: Dict) -> None:
>>>>>>> REPLACE

<<<<<<< SEARCH
    # Residual-output quotients retain dense output-coordinate moments. The
    # optimization-sensitive key row 15 and value rows 20 and 23 additionally
    # use dense moments for their omitted normalized-input coefficients.
    quotient_specifications = []
    qkv_row_specifications = []
=======
    # Residual-output and sensitive QKV quotients retain their verified
    # optimizers. The new coordinate-2 token-position quotient additionally
    # recovers AdamW moments in the original dense embedding columns.
    quotient_specifications = []
    qkv_row_specifications = []
    embedding_quotient_specifications = [
        (model.token_emb, model.pos_emb, 2)
    ]
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
    embedding_parameter_ids = {
        id(parameter)
        for token, position, _ in embedding_quotient_specifications
        for parameter in (token.weight, position.weight)
    }
    optimizer = torch.optim.AdamW(
        [
            parameter
            for parameter in model.parameters()
            if id(parameter) not in quotient_parameter_ids
            and id(parameter) not in qkv_row_parameter_ids
            and id(parameter) not in embedding_parameter_ids
        ],
>>>>>>> REPLACE

<<<<<<< SEARCH
    qkv_row_states = []
    for module, _, target_rows in qkv_row_specifications:
=======
    embedding_quotient_states = []
    for token, position, _ in embedding_quotient_specifications:
        embedding_quotient_states.append(
            {
                "step": 0,
                "token_exp_avg": token.weight.new_zeros(
                    token.weight.shape
                ),
                "token_exp_avg_sq": token.weight.new_zeros(
                    token.weight.shape
                ),
                "position_exp_avg": position.weight.new_zeros(
                    position.weight.shape
                ),
                "position_exp_avg_sq": position.weight.new_zeros(
                    position.weight.shape
                ),
                "dense_token_exp_avg": token.weight.new_zeros(
                    token.num_embeddings
                ),
                "dense_token_exp_avg_sq": token.weight.new_zeros(
                    token.num_embeddings
                ),
                "dense_position_exp_avg": position.weight.new_zeros(
                    position.num_embeddings
                ),
                "dense_position_exp_avg_sq": position.weight.new_zeros(
                    position.num_embeddings
                ),
            }
        )
    qkv_row_states = []
    for module, _, target_rows in qkv_row_specifications:
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
        step_dense_embedding_quotients(
            embedding_quotient_specifications,
            embedding_quotient_states,
            lr_now,
            train_cfg.weight_decay,
        )
        optimizer.step()
>>>>>>> REPLACE