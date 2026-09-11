MECHANISM: Dense-AdamW common-output affine quotient

HYPOTHESIS: Removing every common-output weight direction from the attention projection and final MLP, while projecting ordinary dense-coordinate AdamW updates into the orthonormal quotient, will produce a 1,552-parameter model with at least 99% accuracy.

INTENDED_EDIT: Extend the verified common-bias gauges to all affine columns of both residual-branch output layers, retain dense-coordinate AdamW geometry for the newly gauged weights and final-MLP bias, and reproduce the verified tied-token translation gauge.

EVIDENCE: The 1,573-parameter orthonormal attention-bias quotient achieved 99.97%, and dense-coordinate AdamW raised the final-MLP bias quotient from 98.89% to 99.97% at 1,572 parameters; each removed weight direction likewise contributes only an input-dependent common residual translation erased by downstream LayerNorm.

<<<<<<< SEARCH
        return F.linear(x, weight, fused_bias)


class CausalSelfAttention(nn.Module):
=======
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
        # by every output coordinate become common residual translations and
        # are erased by the downstream LayerNorm.
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
        # Retain all but one query-bias coordinate while applying the verified
        # query, key, and value normalized-input gauges.
        self.qkv = LayerNormGaugedQKV(d_model, self.head_dim)
        self.proj = nn.Linear(d_model, d_model)
=======
        # Retain the verified QKV gauges and quotient all common-output affine
        # directions of the residual projection.
        self.qkv = LayerNormGaugedQKV(d_model, self.head_dim)
        self.proj = OrthonormalCommonOutputGaugedLinear(d_model, d_model)
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.fc1 = LayerNormGaugedLinear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.drop = nn.Dropout(dropout)
=======
        self.fc1 = LayerNormGaugedLinear(d_model, d_ff)
        self.fc2 = OrthonormalCommonOutputGaugedLinear(d_ff, d_model)
        self.drop = nn.Dropout(dropout)
>>>>>>> REPLACE

<<<<<<< SEARCH
class ThreeCoordinateGaugedPositionEmbedding(nn.Module):
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

        # A global embedding translation is invisible to input LayerNorms and
        # contributes only a vocabulary-common output-logit offset.
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


class ThreeCoordinateGaugedPositionEmbedding(nn.Module):
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.cfg = cfg
        self.token_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_emb = ThreeCoordinateGaugedPositionEmbedding(
            cfg.max_seq_len, cfg.d_model
        )
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.ln_f = nn.LayerNorm(cfg.d_model)

        # Weight tying with input embeddings.
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.token_emb.weight

        self.apply(self._init_weights)

        # Preserve initialized token-plus-position inputs. Tied output
        # embeddings turn each transferred shift into a common logit offset.
        with torch.no_grad():
            self.token_emb.weight[:, 0].add_(
                self.pos_emb._init_token_shift[0]
            )
            self.token_emb.weight[:, 3].add_(
                self.pos_emb._init_token_shift[1]
            )
            self.token_emb.weight[:, 7].add_(
                self.pos_emb._init_token_shift[2]
            )
=======
        self.cfg = cfg
        self.token_emb = ScalarGaugedTokenEmbedding(
            cfg.vocab_size, cfg.d_model
        )
        self.pos_emb = ThreeCoordinateGaugedPositionEmbedding(
            cfg.max_seq_len, cfg.d_model
        )
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.ln_f = nn.LayerNorm(cfg.d_model)

        # Reconstruct input and output weights from one shared gauged parameter.
        self.lm_head = TiedGaugedLMHead(self.token_emb)

        self.apply(self._init_weights)

        # Transfer positional anchors before restoring the global tied-token
        # translation gauge.
        self.token_emb.transfer_coordinate_shifts(
            (0, 3, 7), self.pos_emb._init_token_shift
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, ThreeCoordinateGaugedPositionEmbedding):
=======
    @staticmethod
    def _init_weights(module: nn.Module) -> None:
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
        elif isinstance(module, ThreeCoordinateGaugedPositionEmbedding):
>>>>>>> REPLACE

<<<<<<< SEARCH
                module._init_token_shift.copy_(shifts)
        elif isinstance(module, LayerNormGaugedQKV):
=======
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
def save_json(path: Path, obj: Dict) -> None:
=======
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

        # Matrix multiplication handles both affine weights and bias vectors.
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
>>>>>>> REPLACE