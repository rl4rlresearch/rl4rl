MECHANISM: Orthonormal attention-projection common-bias gauge

HYPOTHESIS: Reproducing the verified 1,574-parameter design and representing each attention projection bias in an orthonormal zero-sum basis will yield 1,573 parameters while maintaining at least 99% accuracy.

INTENDED_EDIT: Add the verified QKV, balanced query-bias, positional, and tied-token gauges, then remove the attention projection bias’s downstream-LayerNorm-invariant common direction without privileging a hidden coordinate.

EVIDENCE: The 1,574-parameter reference achieved 99.88% accuracy, while an additional positional gauge failed. The final-MLP anchored-bias experiment motivates testing the distinct attention-projection null direction with an orthonormal centered chart instead of another coordinate anchor.

<<<<<<< SEARCH
        # Balance four query-row gauges across the two attention heads, then
        # gauge the first key row without further constraining either query.
        self.gauged_rows = (0, 1, head_dim, head_dim + 1, d_model)
        self.ungauged_rows = tuple(
            row for row in range(self.out_features) if row not in self.gauged_rows
        )

        # Consume the same constructor RNG stream as the replaced nn.Linear.
        base = nn.Linear(d_model, 3 * d_model)
        retained = self.out_features * d_model - len(self.gauged_rows)
        self.weight = nn.Parameter(base.weight.new_empty(retained))
        self.bias = nn.Parameter(base.bias.new_empty(d_model))
=======
        # Retain the verified balanced query and first-key gauges, then gauge
        # two adjacent value coordinates in the first head.
        self.gauged_rows = (
            0,
            1,
            head_dim,
            head_dim + 1,
            d_model,
            2 * d_model,
            2 * d_model + 1,
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


class OrthonormalCommonBiasGaugedLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # Consume the same constructor RNG stream as the replaced nn.Linear.
        base = nn.Linear(in_features, out_features)
        self.weight = nn.Parameter(
            base.weight.new_empty(out_features, in_features)
        )
        self.bias = nn.Parameter(base.bias.new_empty(out_features - 1))

        # Helmert columns are an orthonormal basis for zero-mean biases.
        basis = torch.zeros(out_features, out_features - 1)
        for column in range(out_features - 1):
            scale = math.sqrt((column + 1) * (column + 2))
            basis[: column + 1, column] = 1.0 / scale
            basis[column + 1, column] = -(column + 1) / scale
        self.register_buffer("bias_basis", basis, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bias = self.bias_basis @ self.bias
        return F.linear(x, self.weight, bias)


class CausalSelfAttention(nn.Module):
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.proj = nn.Linear(d_model, d_model)
=======
        self.proj = OrthonormalCommonBiasGaugedLinear(d_model, d_model)
>>>>>>> REPLACE

<<<<<<< SEARCH
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class TinyDecoderLM(nn.Module):
=======
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


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


class TinyDecoderLM(nn.Module):
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.cfg = cfg
        self.token_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_emb = nn.Embedding(cfg.max_seq_len, cfg.d_model)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.ln_f = nn.LayerNorm(cfg.d_model)

        # Weight tying with input embeddings.
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.token_emb.weight

        self.apply(self._init_weights)
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

        # Reconstruct input and output weights from one shared parameter.
        self.lm_head = TiedGaugedLMHead(self.token_emb)

        self.apply(self._init_weights)

        self.token_emb.transfer_coordinate_shifts(
            (0, 3, 7), self.pos_emb._init_token_shift
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, LayerNormGaugedQKV):
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
        elif isinstance(module, OrthonormalCommonBiasGaugedLinear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            nn.init.zeros_(module.bias)
        elif isinstance(module, LayerNormGaugedQKV):
>>>>>>> REPLACE