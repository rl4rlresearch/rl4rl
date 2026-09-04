MECHANISM: Alternative fifth token–position translation quotient at coordinate 5

HYPOTHESIS: Extending the verified 1,535-parameter design by anchoring position-zero coordinate 5 will produce a 1,534-parameter model with at least 99% accuracy.

INTENDED_EDIT: Reproduce the verified query-row-7 and four-position-anchor design, retain dense-coordinate AdamW for sensitive QKV rows 15, 20, and 23, and add coordinate 5 to the positional translation gauge.

EVIDENCE: The four-anchor design achieved 99.82% at 1,535 parameters. Coordinates 2 and 4 were optimization-sensitive failures, so testing the remaining coordinate 5—while preserving its learned query bias—is the most direct informative one-parameter reduction.

<<<<<<< SEARCH
        # Retain the verified balanced query gauges, gauge every first-head key
        # row plus three adjacent second-head key rows, and gauge every
        # first-head value row.
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
            2 * d_model,
            2 * d_model + 1,
            2 * d_model + 2,
            2 * d_model + 3,
        )
=======
        # Retain the verified query-row-7 design and every key and value
        # gauge. Sensitive rows 15, 20, and 23 use recovered dense-coordinate
        # AdamW moments.
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
        # Flat indices 0, 3, and 7 are position zero's selected coordinates.
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
class FiveCoordinateGaugedPositionEmbedding(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim

        # Consume the same constructor RNG stream as nn.Embedding.
        base = nn.Embedding(num_embeddings, embedding_dim)
        retained = num_embeddings * embedding_dim - 5
        self.weight = nn.Parameter(base.weight.new_empty(retained))
        self.register_buffer(
            "_init_token_shift", base.weight.new_zeros(5), persistent=False
        )

    def forward(self, indices: torch.Tensor) -> torch.Tensor:
        # Anchor position zero at coordinates 0, 1, 3, 5, and 7.
        flat = torch.cat(
            (
                self.weight.new_zeros(2),
                self.weight[:1],
                self.weight.new_zeros(1),
                self.weight[1:2],
                self.weight.new_zeros(1),
                self.weight[2:3],
                self.weight.new_zeros(1),
                self.weight[3:],
            )
        )
        return F.embedding(
            indices, flat.view(self.num_embeddings, self.embedding_dim)
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.pos_emb = ThreeCoordinateGaugedPositionEmbedding(
            cfg.max_seq_len, cfg.d_model
        )
=======
        self.pos_emb = FiveCoordinateGaugedPositionEmbedding(
            cfg.max_seq_len, cfg.d_model
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
        # Transfer positional anchors before restoring the global tied-token
        # translation gauge.
        self.token_emb.transfer_coordinate_shifts(
            (0, 3, 7), self.pos_emb._init_token_shift
        )
=======
        # Transfer positional anchors before restoring the global tied-token
        # translation gauge.
        self.token_emb.transfer_coordinate_shifts(
            (0, 1, 3, 5, 7), self.pos_emb._init_token_shift
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
                        full[0, 3],
                        full[0, 5],
                        full[0, 7],
                    )
                ).clone()
                full[:, 0].sub_(shifts[0])
                full[:, 1].sub_(shifts[1])
                full[:, 3].sub_(shifts[2])
                full[:, 5].sub_(shifts[3])
                full[:, 7].sub_(shifts[4])
                flat = full.flatten()
                module.weight.copy_(
                    torch.cat(
                        (flat[2:3], flat[4:5], flat[6:7], flat[8:])
                    )
                )
                module._init_token_shift.copy_(shifts)
>>>>>>> REPLACE

<<<<<<< SEARCH
        parameter.add_(
            module.output_basis.transpose(0, 1) @ dense_update,
            alpha=-lr / bias_correction1,
        )


def save_json(path: Path, obj: Dict) -> None:
=======
        parameter.add_(
            module.output_basis.transpose(0, 1) @ dense_update,
            alpha=-lr / bias_correction1,
        )


@torch.no_grad()
def step_dense_qkv_row_quotients(
    specifications, states, lr: float, weight_decay: float
) -> None:
    """Apply dense AdamW moments to selected normalized-input row quotients."""
    beta1, beta2, eps = 0.9, 0.999, 1e-8

    for (module, normalization, target_rows), state in zip(
        specifications, states
    ):
        parameter = module.weight
        if parameter.grad is None:
            continue

        row_width = module.in_features - 1
        target_slices = []
        anchored_rows = []
        reduced_grads = []
        for target_row in target_rows:
            target_index = module.gauged_rows.index(target_row)
            start = target_index * row_width
            stop = start + row_width
            target_slices.append((start, stop))
            anchored_rows.append(parameter[start:stop].clone())
            reduced_grads.append(parameter.grad[start:stop])

        anchored_weight = torch.stack(anchored_rows)
        reduced_grad = torch.stack(reduced_grads)
        scales = torch.cat(
            (
                normalization.weight,
                normalization.weight.new_ones(
                    module.in_features - normalization.weight.numel()
                ),
            )
        )
        null_direction = scales.reciprocal()
        last_grad = -(
            reduced_grad * null_direction[:-1]
        ).sum(dim=1, keepdim=True) / null_direction[-1]
        dense_grad = torch.cat((reduced_grad, last_grad), dim=1)

        state["step"] += 1
        step = state["step"]
        bias_correction1 = 1.0 - beta1 ** step
        bias_correction2 = 1.0 - beta2 ** step

        # Preserve reduced-coordinate AdamW for every non-targeted QKV row.
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

        # Update key row 15 and value rows 20 and 23 in eight-coordinate
        # geometry, then project them back into the anchored chart.
        dense_exp_avg = state["dense_exp_avg"]
        dense_exp_avg_sq = state["dense_exp_avg_sq"]
        dense_exp_avg.mul_(beta1).add_(dense_grad, alpha=1.0 - beta1)
        dense_exp_avg_sq.mul_(beta2).addcmul_(
            dense_grad, dense_grad, value=1.0 - beta2
        )
        dense_denom = dense_exp_avg_sq.sqrt().div_(
            math.sqrt(bias_correction2)
        ).add_(eps)
        dense_weight = torch.cat(
            (
                anchored_weight,
                anchored_weight.new_zeros(len(target_rows), 1),
            ),
            dim=1,
        )
        dense_weight.mul_(1.0 - lr * weight_decay)
        dense_weight.add_(
            dense_exp_avg / dense_denom,
            alpha=-lr / bias_correction1,
        )
        dense_weight[:, :-1].sub_(
            dense_weight[:, -1:]
            * null_direction[:-1]
            / null_direction[-1]
        )
        for dense_row, (start, stop) in zip(
            dense_weight, target_slices
        ):
            parameter[start:stop].copy_(dense_row[:-1])


def save_json(path: Path, obj: Dict) -> None:
>>>>>>> REPLACE

<<<<<<< SEARCH
    # Preserve the successful ordinary quotient optimizer for attention bias.
    # Newly gauged weights and the sensitive final-MLP bias use moments in the
    # original dense output coordinates before each update is projected.
    quotient_specifications = []
    for block in model.blocks:
        quotient_specifications.extend(
            (
                (block.attn.proj, "weight"),
                (block.mlp.fc2, "weight"),
                (block.mlp.fc2, "bias"),
            )
        )
    quotient_parameter_ids = {
        id(getattr(module, parameter_name))
        for module, parameter_name in quotient_specifications
    }
    optimizer = torch.optim.AdamW(
        [
            parameter
            for parameter in model.parameters()
            if id(parameter) not in quotient_parameter_ids
        ],
        lr=train_cfg.lr,
        weight_decay=train_cfg.weight_decay,
    )
    quotient_states = []
=======
    # Residual-output quotients retain dense output-coordinate moments. The
    # optimization-sensitive key row 15 and value rows 20 and 23 additionally
    # use dense moments for their omitted normalized-input coefficients.
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
>>>>>>> REPLACE

<<<<<<< SEARCH
        optimizer.step()
        step_dense_output_quotients(
=======
        step_dense_qkv_row_quotients(
            qkv_row_specifications,
            qkv_row_states,
            lr_now,
            train_cfg.weight_decay,
        )
        optimizer.step()
        step_dense_output_quotients(
>>>>>>> REPLACE