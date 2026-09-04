MECHANISM: Dense-AdamW token–position translation quotient

HYPOTHESIS: Starting from the verified 1,533-parameter all-row-QKV design, anchoring position-zero coordinate 5 and optimizing its omitted token–position shift in dense coordinates will produce a 1,532-parameter model with at least 99% accuracy.

INTENDED_EDIT: Reproduce the verified five-position-anchor and all-row-QKV gauges, add coordinate 5 as a sixth positional anchor, and use dense-coordinate AdamW for that exact token–position translation quotient.

EVIDENCE: Reference Design 3 achieved 99.6% at 1,533 parameters. The direct coordinate-5 anchor reached 95.94%, while dense-coordinate optimization previously rescued sensitive exact QKV quotients, motivating dense optimization of this still-untested positional quotient.

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
=======
        # Gauge every QKV row. Query row 6 and sensitive rows 15, 20, and 23
        # use recovered dense-coordinate AdamW moments.
        self.gauged_rows = (
            0,
            1,
            2,
            3,
            head_dim,
            head_dim + 1,
            head_dim + 2,
            head_dim + 3,
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
class SixCoordinateGaugedPositionEmbedding(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.dense_coordinate = 5
        self.anchor_count = 6

        # Consume the same constructor RNG stream as nn.Embedding.
        base = nn.Embedding(num_embeddings, embedding_dim)
        retained = num_embeddings * embedding_dim - self.anchor_count
        self.weight = nn.Parameter(base.weight.new_empty(retained))
        self.register_buffer(
            "_init_token_shift",
            base.weight.new_zeros(self.anchor_count),
            persistent=False,
        )

    def forward(self, indices: torch.Tensor) -> torch.Tensor:
        # Anchor position zero at coordinates 0, 1, 3, 5, 6, and 7.
        flat = torch.cat(
            (
                self.weight.new_zeros(2),
                self.weight[:1],
                self.weight.new_zeros(1),
                self.weight[1:2],
                self.weight.new_zeros(3),
                self.weight[2:],
            )
        )
        return F.embedding(
            indices, flat.view(self.num_embeddings, self.embedding_dim)
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.pos_emb = FourCoordinateGaugedPositionEmbedding(
            cfg.max_seq_len, cfg.d_model
        )
=======
        self.pos_emb = SixCoordinateGaugedPositionEmbedding(
            cfg.max_seq_len, cfg.d_model
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.token_emb.transfer_coordinate_shifts(
            (0, 1, 3, 7), self.pos_emb._init_token_shift
        )
=======
        self.token_emb.transfer_coordinate_shifts(
            (0, 1, 3, 5, 6, 7), self.pos_emb._init_token_shift
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
        elif isinstance(module, SixCoordinateGaugedPositionEmbedding):
            with torch.no_grad():
                full = module.weight.new_empty(
                    module.num_embeddings, module.embedding_dim
                )
                nn.init.normal_(full, mean=0.0, std=0.02)
                shifts = torch.stack(
                    (
                        full[0, 0],
                        full[0, 1],
                        full[0, 3],
                        full[0, 5],
                        full[0, 6],
                        full[0, 7],
                    )
                ).clone()
                full[:, 0].sub_(shifts[0])
                full[:, 1].sub_(shifts[1])
                full[:, 3].sub_(shifts[2])
                full[:, 5].sub_(shifts[3])
                full[:, 6].sub_(shifts[4])
                full[:, 7].sub_(shifts[5])
                flat = full.flatten()
                module.weight.copy_(
                    torch.cat((flat[2:3], flat[4:5], flat[8:]))
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
def step_dense_position_translation_quotient(
    token_embedding,
    position_embedding,
    state,
    lr: float,
    weight_decay: float,
) -> None:
    """Apply dense AdamW to one exact token-position translation quotient."""
    token_parameter = token_embedding.weight
    position_parameter = position_embedding.weight
    if token_parameter.grad is None or position_parameter.grad is None:
        return

    beta1, beta2, eps = 0.9, 0.999, 1e-8
    coordinate = position_embedding.dense_coordinate
    token_indices = torch.arange(
        coordinate - 1,
        token_parameter.numel(),
        token_embedding.embedding_dim,
        device=token_parameter.device,
    )
    position_indices = torch.arange(
        position_embedding.embedding_dim
        + coordinate
        - position_embedding.anchor_count,
        position_parameter.numel(),
        position_embedding.embedding_dim,
        device=position_parameter.device,
    )

    anchored_token = token_parameter[token_indices].clone()
    anchored_position = position_parameter[position_indices].clone()
    token_gradient = token_parameter.grad[token_indices]
    position_gradient = position_parameter.grad[position_indices]

    # Translation invariance determines the gradient of the omitted
    # position-zero coefficient.
    omitted_position_gradient = (
        token_gradient.sum() - position_gradient.sum()
    )
    dense_gradient = torch.cat(
        (
            token_gradient,
            omitted_position_gradient.reshape(1),
            position_gradient,
        )
    )

    state["step"] += 1
    step = state["step"]
    bias_correction1 = 1.0 - beta1 ** step
    bias_correction2 = 1.0 - beta2 ** step

    # Preserve ordinary reduced-coordinate AdamW for every other retained
    # token and position coefficient.
    for parameter, prefix in (
        (token_parameter, "token"),
        (position_parameter, "position"),
    ):
        exp_avg = state[f"{prefix}_exp_avg"]
        exp_avg_sq = state[f"{prefix}_exp_avg_sq"]
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

    dense_exp_avg = state["dense_exp_avg"]
    dense_exp_avg_sq = state["dense_exp_avg_sq"]
    dense_exp_avg.mul_(beta1).add_(dense_gradient, alpha=1.0 - beta1)
    dense_exp_avg_sq.mul_(beta2).addcmul_(
        dense_gradient, dense_gradient, value=1.0 - beta2
    )
    dense_denom = dense_exp_avg_sq.sqrt().div_(
        math.sqrt(bias_correction2)
    ).add_(eps)

    dense_weight = torch.cat(
        (
            anchored_token,
            anchored_position.new_zeros(1),
            anchored_position,
        )
    )
    dense_weight.mul_(1.0 - lr * weight_decay)
    dense_weight.add_(
        dense_exp_avg / dense_denom,
        alpha=-lr / bias_correction1,
    )

    token_count = token_embedding.num_embeddings
    position_zero = dense_weight[token_count].clone()
    gauged_token = dense_weight[:token_count] + position_zero
    gauged_position = dense_weight[token_count + 1 :] - position_zero
    token_parameter.index_copy_(0, token_indices, gauged_token)
    position_parameter.index_copy_(
        0, position_indices, gauged_position
    )


def save_json(path: Path, obj: Dict) -> None:
>>>>>>> REPLACE

<<<<<<< SEARCH
    # Residual-output quotients retain dense output-coordinate moments. The
    # optimization-sensitive key row 15 and value rows 20 and 23 additionally
    # use dense moments for their omitted normalized-input coefficients.
=======
    # Residual-output quotients retain dense output-coordinate moments. Query
    # row 6 and sensitive rows 15, 20, and 23 use dense QKV moments, while the
    # new position-coordinate-5 quotient uses dense token-position moments.
>>>>>>> REPLACE

<<<<<<< SEARCH
                (
                    2 * qkv.in_features - 1,
                    2 * qkv.in_features + block.attn.head_dim,
                    3 * qkv.in_features - 1,
                ),
=======
                (
                    block.attn.head_dim + 2,
                    2 * qkv.in_features - 1,
                    2 * qkv.in_features + block.attn.head_dim,
                    3 * qkv.in_features - 1,
                ),
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
    translation_parameter_ids = {
        id(model.token_emb.weight),
        id(model.pos_emb.weight),
    }
    optimizer = torch.optim.AdamW(
        [
            parameter
            for parameter in model.parameters()
            if id(parameter) not in quotient_parameter_ids
            and id(parameter) not in qkv_row_parameter_ids
            and id(parameter) not in translation_parameter_ids
        ],
>>>>>>> REPLACE

<<<<<<< SEARCH
                "dense_exp_avg_sq": parameter.new_zeros(
                    len(target_rows), module.in_features
                ),
            }
        )
    quotient_states = []
=======
                "dense_exp_avg_sq": parameter.new_zeros(
                    len(target_rows), module.in_features
                ),
            }
        )
    translation_state = {
        "step": 0,
        "token_exp_avg": model.token_emb.weight.new_zeros(
            model.token_emb.weight.shape
        ),
        "token_exp_avg_sq": model.token_emb.weight.new_zeros(
            model.token_emb.weight.shape
        ),
        "position_exp_avg": model.pos_emb.weight.new_zeros(
            model.pos_emb.weight.shape
        ),
        "position_exp_avg_sq": model.pos_emb.weight.new_zeros(
            model.pos_emb.weight.shape
        ),
        "dense_exp_avg": model.token_emb.weight.new_zeros(
            model.token_emb.num_embeddings + model.pos_emb.num_embeddings
        ),
        "dense_exp_avg_sq": model.token_emb.weight.new_zeros(
            model.token_emb.num_embeddings + model.pos_emb.num_embeddings
        ),
    }
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
        step_dense_position_translation_quotient(
            model.token_emb,
            model.pos_emb,
            translation_state,
            lr_now,
            train_cfg.weight_decay,
        )
        optimizer.step()
>>>>>>> REPLACE