MECHANISM: Dense-AdamW token-position translation quotient

HYPOTHESIS: Extending the verified 1,552-parameter common-output affine quotient with a fourth positional translation gauge at coordinate 4, while projecting ordinary dense-coordinate AdamW updates back into the anchored chart, will yield 1,551 parameters and at least 99% accuracy.

INTENDED_EDIT: Reproduce the verified balanced query-bias, tied-token, three-position, and common-output affine gauges; additionally anchor positional coordinate 4 and train its coupled token-position quotient with a recovered dense AdamW moment for the omitted coefficient.

EVIDENCE: The dense-coordinate optimizer rescued the exact final-MLP quotient from 98.89% to 99.97%, and the resulting common-output extension achieved 99.8% at 1,552 parameters; this directly motivates retesting the exact coordinate-4 token-position gauge whose ordinary and centered-coordinate optimizers previously failed.

<<<<<<< SEARCH
        self.weight = nn.Parameter(base.weight.new_empty(retained))
        self.bias = nn.Parameter(base.bias.new_empty(d_model))
=======
        self.weight = nn.Parameter(base.weight.new_empty(retained))
        # Fix local query coordinate 2 in both attention heads; key and value
        # biases remain absent through their exact attention gauges.
        self.fixed_query_biases = (2, head_dim + 2)
        self.bias = nn.Parameter(
            base.bias.new_empty(d_model - len(self.fixed_query_biases))
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
        fused_bias = torch.cat(
            (
                self.bias,
                self.bias.new_zeros(self.in_features),
                self.bias.new_zeros(self.in_features),
            )
        )
        return F.linear(x, weight, fused_bias)


class CausalSelfAttention(nn.Module):
=======
        query_bias_parts = []
        bias_index = 0
        for coordinate in range(self.in_features):
            if coordinate in self.fixed_query_biases:
                query_bias_parts.append(self.bias.new_zeros(()))
            else:
                query_bias_parts.append(self.bias[bias_index])
                bias_index += 1
        query_bias = torch.stack(query_bias_parts)
        fused_bias = torch.cat(
            (
                query_bias,
                self.bias.new_zeros(self.in_features),
                self.bias.new_zeros(self.in_features),
            )
        )
        return F.linear(x, weight, fused_bias)


class OrthonormalCommonOutputGaugedLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # Consume the same constructor RNG stream as the replaced nn.Linear.
        base = nn.Linear(in_features, out_features)
        self.weight = nn.Parameter(
            base.weight.new_empty(out_features - 1, in_features)
        )
        self.bias = nn.Parameter(base.bias.new_empty(out_features - 1))

        # Helmert columns span the zero-mean output subspace. Components shared
        # by every output coordinate are erased by the downstream LayerNorm.
        basis = torch.zeros(out_features, out_features - 1)
        for column in range(out_features - 1):
            scale = math.sqrt((column + 1) * (column + 2))
            basis[: column + 1, column] = 1.0 / scale
            basis[column + 1, column] = -(column + 1) / scale
        self.register_buffer("output_basis", basis, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = self.output_basis @ self.weight
        bias = self.output_basis @ self.bias
        return F.linear(x, weight, bias)


class CausalSelfAttention(nn.Module):
>>>>>>> REPLACE

<<<<<<< SEARCH
        # Retain query bias while applying the verified query, key, and value
        # normalized-input gauges.
        self.qkv = LayerNormGaugedQKV(d_model, self.head_dim)
        self.proj = nn.Linear(d_model, d_model)
=======
        # Retain the verified QKV gauges and quotient every common-output
        # affine direction of the residual projection.
        self.qkv = LayerNormGaugedQKV(d_model, self.head_dim)
        self.proj = OrthonormalCommonOutputGaugedLinear(d_model, d_model)
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.fc1 = LayerNormGaugedLinear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
=======
        self.fc1 = LayerNormGaugedLinear(d_model, d_ff)
        self.fc2 = OrthonormalCommonOutputGaugedLinear(d_ff, d_model)
>>>>>>> REPLACE

<<<<<<< SEARCH
class TwoCoordinateGaugedPositionEmbedding(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim

        # Consume the same constructor RNG stream as nn.Embedding.
        base = nn.Embedding(num_embeddings, embedding_dim)
        retained = num_embeddings * embedding_dim - 2
        self.weight = nn.Parameter(base.weight.new_empty(retained))
        self.register_buffer(
            "_init_token_shift", base.weight.new_zeros(2), persistent=False
        )

    def forward(self, indices: torch.Tensor) -> torch.Tensor:
        # Flat indices 0 and 3 are position zero's coordinates 0 and 3.
        flat = torch.cat(
            (
                self.weight.new_zeros(1),
                self.weight[:2],
                self.weight.new_zeros(1),
                self.weight[2:],
            )
        )
        return F.embedding(
            indices, flat.view(self.num_embeddings, self.embedding_dim)
        )
=======
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

        # A common scalar embedding translation is invisible to the input
        # LayerNorms and contributes only a common output-logit offset.
        anchor = full[0, 0].clone()
        full.sub_(anchor)
        self.weight.copy_(full.flatten()[1:])


class TiedGaugedLMHead(nn.Module):
    def __init__(self, embedding: ScalarGaugedTokenEmbedding):
        super().__init__()
        self.in_features = embedding.embedding_dim
        self.out_features = embedding.num_embeddings

        # Consume the constructor RNG stream of the replaced tied nn.Linear.
        base = nn.Linear(self.in_features, self.out_features, bias=False)
        object.__setattr__(self, "_embedding", embedding)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self._embedding.dense_weight())


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
        # Position zero anchors coordinates 0, 3, 4, and 7.
        flat = torch.cat(
            (
                self.weight.new_zeros(1),
                self.weight[:2],
                self.weight.new_zeros(2),
                self.weight[2:4],
                self.weight.new_zeros(1),
                self.weight[4:],
            )
        )
        return F.embedding(
            indices, flat.view(self.num_embeddings, self.embedding_dim)
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.cfg = cfg
        self.token_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_emb = TwoCoordinateGaugedPositionEmbedding(
            cfg.max_seq_len, cfg.d_model
        )
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.ln_f = nn.LayerNorm(cfg.d_model)

        # Weight tying with input embeddings.
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.token_emb.weight

        self.apply(self._init_weights)

        # Preserve every initialized input embedding sum. Because the output
        # embedding is tied, each shift changes logits only by a common offset.
        with torch.no_grad():
            self.token_emb.weight[:, 0].add_(
                self.pos_emb._init_token_shift[0]
            )
            self.token_emb.weight[:, 3].add_(
                self.pos_emb._init_token_shift[1]
            )
=======
        self.cfg = cfg
        self.token_emb = ScalarGaugedTokenEmbedding(
            cfg.vocab_size, cfg.d_model
        )
        self.pos_emb = FourCoordinateGaugedPositionEmbedding(
            cfg.max_seq_len, cfg.d_model
        )
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.ln_f = nn.LayerNorm(cfg.d_model)

        # Reconstruct input and output weights from one shared gauged parameter.
        self.lm_head = TiedGaugedLMHead(self.token_emb)

        self.apply(self._init_weights)

        # Transfer all positional anchors before restoring the tied-token
        # scalar translation gauge.
        self.token_emb.transfer_coordinate_shifts(
            (0, 3, 4, 7), self.pos_emb._init_token_shift
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
        if isinstance(module, TwoCoordinateGaugedPositionEmbedding):
            with torch.no_grad():
                full = module.weight.new_empty(
                    module.num_embeddings, module.embedding_dim
                )
                nn.init.normal_(full, mean=0.0, std=0.02)
                shifts = torch.stack((full[0, 0], full[0, 3])).clone()
                full[:, 0].sub_(shifts[0])
                full[:, 3].sub_(shifts[1])
                flat = full.flatten()
                module.weight.copy_(
                    torch.cat((flat[1:3], flat[4:]))
                )
                module._init_token_shift.copy_(shifts)
        elif isinstance(module, LayerNormGaugedQKV):
=======
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
        elif isinstance(module, FourCoordinateGaugedPositionEmbedding):
            with torch.no_grad():
                full = module.weight.new_empty(
                    module.num_embeddings, module.embedding_dim
                )
                nn.init.normal_(full, mean=0.0, std=0.02)
                shifts = torch.stack(
                    (full[0, 0], full[0, 3], full[0, 4], full[0, 7])
                ).clone()
                full[:, 0].sub_(shifts[0])
                full[:, 3].sub_(shifts[1])
                full[:, 4].sub_(shifts[2])
                full[:, 7].sub_(shifts[3])
                flat = full.flatten()
                module.weight.copy_(
                    torch.cat((flat[1:3], flat[5:7], flat[8:]))
                )
                module._init_token_shift.copy_(shifts)
        elif isinstance(module, OrthonormalCommonOutputGaugedLinear):
            with torch.no_grad():
                full = module.weight.new_empty(
                    module.out_features, module.in_features
                )
                nn.init.normal_(full, mean=0.0, std=0.02)
                module.weight.copy_(
                    module.output_basis.transpose(0, 1) @ full
                )
                nn.init.zeros_(module.bias)
        elif isinstance(module, LayerNormGaugedQKV):
>>>>>>> REPLACE

<<<<<<< SEARCH
    return min_lr + (base_lr - min_lr) * cosine


def save_json(path: Path, obj: Dict) -> None:
=======
    return min_lr + (base_lr - min_lr) * cosine


@torch.no_grad()
def step_dense_output_quotients(
    specifications, states, lr: float, weight_decay: float
) -> None:
    """Apply dense-coordinate AdamW and retain its zero-mean quotient."""
    beta1, beta2, eps = 0.9, 0.999, 1e-8

    for (module, parameter_name), state in zip(specifications, states):
        parameter = getattr(module, parameter_name)
        if parameter.grad is None:
            continue

        dense_grad = module.output_basis @ parameter.grad
        state["step"] += 1
        step = state["step"]
        exp_avg = state["exp_avg"]
        exp_avg_sq = state["exp_avg_sq"]

        exp_avg.mul_(beta1).add_(dense_grad, alpha=1.0 - beta1)
        exp_avg_sq.mul_(beta2).addcmul_(
            dense_grad, dense_grad, value=1.0 - beta2
        )

        bias_correction1 = 1.0 - beta1 ** step
        bias_correction2 = 1.0 - beta2 ** step
        denom = exp_avg_sq.sqrt().div_(
            math.sqrt(bias_correction2)
        ).add_(eps)
        dense_update = exp_avg / denom

        parameter.mul_(1.0 - lr * weight_decay)
        parameter.add_(
            module.output_basis.transpose(0, 1) @ dense_update,
            alpha=-lr / bias_correction1,
        )


@torch.no_grad()
def step_embedding_translation_quotient(
    token_embedding,
    position_embedding,
    state,
    coordinate: int,
    lr: float,
    weight_decay: float,
) -> None:
    """Apply chart AdamW, recovering the omitted dense positional moment."""
    beta1, beta2, eps = 0.9, 0.999, 1e-8
    token_parameter = token_embedding.weight
    position_parameter = position_embedding.weight
    if token_parameter.grad is None or position_parameter.grad is None:
        return

    token_indices = (
        torch.arange(
            token_embedding.num_embeddings,
            device=token_parameter.device,
        )
        * token_embedding.embedding_dim
        + coordinate
        - 1
    )
    position_indices = (
        torch.arange(
            1,
            position_embedding.num_embeddings,
            device=position_parameter.device,
        )
        * position_embedding.embedding_dim
    )

    token_grad = token_parameter.grad
    position_grad = position_parameter.grad
    missing_grad = (
        token_grad[token_indices].sum()
        - position_grad[position_indices].sum()
    )

    state["step"] += 1
    step = state["step"]
    bias_correction1 = 1.0 - beta1 ** step
    bias_correction2 = 1.0 - beta2 ** step

    for parameter, grad, exp_avg, exp_avg_sq in (
        (
            token_parameter,
            token_grad,
            state["token_exp_avg"],
            state["token_exp_avg_sq"],
        ),
        (
            position_parameter,
            position_grad,
            state["position_exp_avg"],
            state["position_exp_avg_sq"],
        ),
    ):
        exp_avg.mul_(beta1).add_(grad, alpha=1.0 - beta1)
        exp_avg_sq.mul_(beta2).addcmul_(
            grad, grad, value=1.0 - beta2
        )
        denom = exp_avg_sq.sqrt().div_(
            math.sqrt(bias_correction2)
        ).add_(eps)
        parameter.mul_(1.0 - lr * weight_decay)
        parameter.addcdiv_(
            exp_avg, denom, value=-lr / bias_correction1
        )

    missing_exp_avg = state["missing_exp_avg"]
    missing_exp_avg_sq = state["missing_exp_avg_sq"]
    missing_exp_avg.mul_(beta1).add_(
        missing_grad, alpha=1.0 - beta1
    )
    missing_exp_avg_sq.mul_(beta2).add_(
        missing_grad.square(), alpha=1.0 - beta2
    )
    missing_denom = (
        missing_exp_avg_sq.sqrt()
        / math.sqrt(bias_correction2)
        + eps
    )
    missing_update = (
        -lr * missing_exp_avg / (bias_correction1 * missing_denom)
    )

    # Re-anchor the updated dense position-zero coefficient. Adding the same
    # shift to every token and subtracting it from every position preserves
    # all input states and changes output logits only by a common offset.
    token_parameter[token_indices].add_(missing_update)
    position_parameter[position_indices].sub_(missing_update)


def save_json(path: Path, obj: Dict) -> None:
>>>>>>> REPLACE

<<<<<<< SEARCH
    model = TinyDecoderLM(model_cfg).to(device)
    params = count_parameters(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=train_cfg.lr, weight_decay=train_cfg.weight_decay)

    sampler = TrainBatchSampler(train_cfg.batch_size, train_cfg.seed + 1337, reserved_hashes)
=======
    model = TinyDecoderLM(model_cfg).to(device)
    params = count_parameters(model)

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
    embedding_parameter_ids = {
        id(model.token_emb.weight),
        id(model.pos_emb.weight),
    }
    optimizer = torch.optim.AdamW(
        [
            parameter
            for parameter in model.parameters()
            if id(parameter)
            not in quotient_parameter_ids | embedding_parameter_ids
        ],
        lr=train_cfg.lr,
        weight_decay=train_cfg.weight_decay,
    )
    quotient_states = []
    for module, parameter_name in quotient_specifications:
        dense_shape = (
            (module.out_features, module.in_features)
            if parameter_name == "weight"
            else (module.out_features,)
        )
        quotient_states.append(
            {
                "step": 0,
                "exp_avg": getattr(module, parameter_name).new_zeros(
                    dense_shape
                ),
                "exp_avg_sq": getattr(module, parameter_name).new_zeros(
                    dense_shape
                ),
            }
        )
    embedding_quotient_state = {
        "step": 0,
        "token_exp_avg": torch.zeros_like(model.token_emb.weight),
        "token_exp_avg_sq": torch.zeros_like(model.token_emb.weight),
        "position_exp_avg": torch.zeros_like(model.pos_emb.weight),
        "position_exp_avg_sq": torch.zeros_like(model.pos_emb.weight),
        "missing_exp_avg": model.pos_emb.weight.new_zeros(()),
        "missing_exp_avg_sq": model.pos_emb.weight.new_zeros(()),
    }

    sampler = TrainBatchSampler(train_cfg.batch_size, train_cfg.seed + 1337, reserved_hashes)
>>>>>>> REPLACE

<<<<<<< SEARCH
        _, loss = model(x, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if train_cfg.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg.grad_clip)
        optimizer.step()
=======
        _, loss = model(x, y)
        model.zero_grad(set_to_none=True)
        loss.backward()
        if train_cfg.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg.grad_clip)
        optimizer.step()
        step_dense_output_quotients(
            quotient_specifications,
            quotient_states,
            lr_now,
            train_cfg.weight_decay,
        )
        step_embedding_translation_quotient(
            model.token_emb,
            model.pos_emb,
            embedding_quotient_state,
            4,
            lr_now,
            train_cfg.weight_decay,
        )
>>>>>>> REPLACE