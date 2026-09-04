MECHANISM: Orthonormal final-residual bias quotient

HYPOTHESIS: Combining the qualified 1,628-parameter design with removal of the final MLP bias’s LayerNorm-invisible common mode will produce a 1,627-parameter model with at least 99% accuracy.

INTENDED_EDIT: Apply the verified one-coordinate embedding quotient, omit all key biases and value biases in the qualified 4:3 layout, and represent the eight-coordinate final MLP bias using seven orthonormal zero-mean coordinates.

EVIDENCE: The qualified embedding-plus-key-bias design achieved 99.95% at 1,628 parameters. The failed asymmetric attention-output quotient motivates testing the distinct final-MLP residual bias immediately before `ln_f`, using an orthonormal basis to preserve balanced optimization geometry.

<<<<<<< SEARCH
class CompactQKV(nn.Module):
    """QKV projection omitting seven key biases and three value biases per head."""

    def __init__(self, linear: nn.Linear, key_start: int, second_head_start: int):
        super().__init__()
        self.key_start = key_start
        self.second_head_start = second_head_start
        self.value_start = 2 * key_start
        self.head_dim = second_head_start - key_start
        self.weight = linear.weight
        compact_bias = torch.cat(
            (
                linear.bias[:key_start],
                linear.bias[second_head_start + 3 : self.value_start],
                linear.bias[self.value_start + 3 : self.value_start + self.head_dim],
                linear.bias[self.value_start + self.head_dim + 3 :],
            )
        )
        self.bias = nn.Parameter(compact_bias.detach().clone())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        retained_key_end = self.key_start + 1
        first_value_end = retained_key_end + self.head_dim - 3
        full_bias = torch.cat(
            (
                self.bias[: self.key_start],
                self.bias.new_zeros(7),
                self.bias[self.key_start : retained_key_end],
                self.bias.new_zeros(3),
                self.bias[retained_key_end:first_value_end],
                self.bias.new_zeros(3),
                self.bias[first_value_end:],
            )
        )
        return F.linear(x, self.weight, full_bias)
=======
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


class CompactFinalResidualLinear(nn.Module):
    """Linear map whose final-residual bias has no common-mode coordinate."""

    def __init__(self, linear: nn.Linear):
        super().__init__()
        self.weight = linear.weight
        out_features = linear.out_features
        basis = linear.bias.new_zeros((out_features, out_features - 1))
        for column in range(out_features - 1):
            scale = math.sqrt((column + 1) * (column + 2))
            basis[: column + 1, column] = 1.0 / scale
            basis[column + 1, column] = -(column + 1) / scale
        self.register_buffer("bias_basis", basis, persistent=False)

        centered_bias = linear.bias.detach() - linear.bias.detach().mean()
        compact_bias = basis.transpose(0, 1) @ centered_bias
        self.bias = nn.Parameter(compact_bias.clone())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight, self.bias_basis @ self.bias)


class CompactQKV(nn.Module):
    """QKV projection omitting all key biases and value biases in a 4:3 layout."""

    def __init__(self, linear: nn.Linear, key_start: int, second_head_start: int):
        super().__init__()
        self.key_start = key_start
        self.value_start = 2 * key_start
        self.head_dim = second_head_start - key_start
        self.weight = linear.weight
        compact_bias = torch.cat(
            (
                linear.bias[:key_start],
                linear.bias[self.value_start + self.head_dim + 3 :],
            )
        )
        self.bias = nn.Parameter(compact_bias.detach().clone())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        full_bias = torch.cat(
            (
                self.bias[: self.key_start],
                self.bias.new_zeros(self.key_start),
                self.bias.new_zeros(self.head_dim),
                self.bias.new_zeros(3),
                self.bias[self.key_start :],
            )
        )
        return F.linear(x, self.weight, full_bias)
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.apply(self._init_weights)

        # Preserve every baseline initialization draw before removing seven
        # key biases (4:3) and three value biases from each head.
        for block in self.blocks:
            block.attn.qkv = CompactQKV(
                block.attn.qkv,
                cfg.d_model,
                cfg.d_model + block.attn.head_dim,
            )
=======
        self.apply(self._init_weights)

        # Fix one exact token/position translation gauge while preserving the
        # initialized hidden inputs and output probabilities.
        with torch.no_grad():
            offset = self.pos_emb.weight[0, 0].detach().clone()
            self.token_emb.weight[:, 0].add_(offset)
            self.pos_emb.weight[:, 0].sub_(offset)
        self.pos_emb = CompactPositionEmbedding(self.pos_emb)

        # Preserve every baseline initialization draw before removing all key
        # biases, value biases in a 4:3 layout, and the final residual bias's
        # LayerNorm-invisible common mode.
        for block in self.blocks:
            block.mlp.fc2 = CompactFinalResidualLinear(block.mlp.fc2)
            block.attn.qkv = CompactQKV(
                block.attn.qkv,
                cfg.d_model,
                cfg.d_model + block.attn.head_dim,
            )
>>>>>>> REPLACE