MECHANISM: Orthonormal residual-bias common-mode quotient

HYPOTHESIS: Extending the verified 1,622-parameter design with an orthonormal seven-dimensional parameterization of `fc2` bias will produce 1,621 parameters and maintain at least 99% accuracy, because it removes only the uniform residual offset canceled by the final LayerNorm while preserving initialization and well-conditioned quotient geometry.

INTENDED_EDIT: Reproduce the qualified compact positional, shared-query, compact-`ln2`, and shared value/projection-bias design, then express `fc2.bias` in an orthonormal zero-mean basis.

EVIDENCE: The shared value/projection design achieved 99.92% at 1,622 parameters. A prior final-coordinate `fc2` gauge fix collapsed to 26.13%, motivating an orthonormal quotient that preserves the exact functional symmetry without the asymmetric coordinate parameterization.

<<<<<<< SEARCH
        y = self.resid_drop(y)
        return y


class MLP(nn.Module):
=======
        y = self.resid_drop(y)
        return y


class CompactPositionEmbedding(nn.Module):
    """Positional embedding with one token-position translation gauge fixed."""

    def __init__(self, embedding: nn.Embedding):
        super().__init__()
        self.num_embeddings = embedding.num_embeddings
        self.embedding_dim = embedding.embedding_dim
        self.weight = nn.Parameter(embedding.weight.detach().reshape(-1)[1:].clone())

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        full_weight = torch.cat((self.weight.new_zeros(1), self.weight))
        return F.embedding(
            idx,
            full_weight.view(self.num_embeddings, self.embedding_dim),
        )


class CompactQKV(nn.Module):
    """QKV projection with shared head-0 queries and one retained value bias."""

    def __init__(self, linear: nn.Linear, key_start: int, second_head_start: int):
        super().__init__()
        self.key_start = key_start
        self.value_start = 2 * key_start
        self.head_dim = second_head_start - key_start
        self.weight = linear.weight
        query_bias = torch.cat(
            (
                linear.bias[:1],
                linear.bias[self.head_dim : key_start - 1],
            )
        )
        self.query_bias = nn.Parameter(query_bias.detach().clone())
        self.value_bias = nn.Parameter(
            linear.bias[self.value_start + self.head_dim + 3 :].detach().clone()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        full_bias = torch.cat(
            (
                self.query_bias[:1].expand(self.head_dim - 2),
                self.query_bias.new_zeros(2),
                self.query_bias[1:],
                self.query_bias.new_zeros(1),
                self.query_bias.new_zeros(self.key_start),
                self.query_bias.new_zeros(self.head_dim),
                self.query_bias.new_zeros(3),
                self.value_bias,
            )
        )
        return F.linear(x, self.weight, full_bias)


class CompactSharedProjection(nn.Module):
    """Attention projection sharing its final bias with the retained value bias."""

    def __init__(self, linear: nn.Linear, shared_bias: nn.Parameter):
        super().__init__()
        self.weight = linear.weight
        self.bias = nn.Parameter(linear.bias[:-1].detach().clone())
        self.shared_bias = shared_bias

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        full_bias = torch.cat((self.bias, self.shared_bias))
        return F.linear(x, self.weight, full_bias)


class CompactLayerNormBias(nn.Module):
    """LayerNorm with one downstream-linear-absorbed bias coordinate fixed."""

    def __init__(self, layer_norm: nn.LayerNorm):
        super().__init__()
        self.normalized_shape = layer_norm.normalized_shape
        self.eps = layer_norm.eps
        self.weight = layer_norm.weight
        self.bias = nn.Parameter(layer_norm.bias[:-1].detach().clone())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        full_bias = torch.cat((self.bias, self.bias.new_zeros(1)))
        return F.layer_norm(
            x,
            self.normalized_shape,
            self.weight,
            full_bias,
            self.eps,
        )


class CompactResidualLinear(nn.Module):
    """Linear map whose output bias omits the final-LayerNorm-null uniform mode."""

    def __init__(self, linear: nn.Linear):
        super().__init__()
        self.weight = linear.weight
        out_features = linear.out_features
        basis = linear.weight.new_zeros(out_features, out_features - 1)
        for column in range(out_features - 1):
            scale = 1.0 / math.sqrt((column + 1) * (column + 2))
            basis[: column + 1, column] = scale
            basis[column + 1, column] = -(column + 1) * scale
        compact_bias = basis.transpose(0, 1) @ linear.bias.detach()
        self.bias = nn.Parameter(compact_bias.clone())
        self.register_buffer("bias_basis", basis, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        full_bias = self.bias_basis @ self.bias
        return F.linear(x, self.weight, full_bias)


class MLP(nn.Module):
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.ln2 = nn.LayerNorm(cfg.d_model)
=======
        self.ln2 = CompactLayerNormBias(nn.LayerNorm(cfg.d_model))
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.apply(self._init_weights)

    @staticmethod
=======
        self.apply(self._init_weights)

        # Fix one exact token/position translation gauge while preserving the
        # initialized hidden inputs and output probabilities.
        with torch.no_grad():
            offset = self.pos_emb.weight[0, 0].detach().clone()
            self.token_emb.weight[:, 0].add_(offset)
            self.pos_emb.weight[:, 0].sub_(offset)
        self.pos_emb = CompactPositionEmbedding(self.pos_emb)

        # Retain the qualified attention layout and remove the uniform fc2
        # residual-bias mode in an orthonormal quotient basis.
        for block in self.blocks:
            compact_qkv = CompactQKV(
                block.attn.qkv,
                cfg.d_model,
                cfg.d_model + block.attn.head_dim,
            )
            block.attn.qkv = compact_qkv
            block.attn.proj = CompactSharedProjection(
                block.attn.proj,
                compact_qkv.value_bias,
            )
            block.mlp.fc2 = CompactResidualLinear(block.mlp.fc2)

    @staticmethod
>>>>>>> REPLACE