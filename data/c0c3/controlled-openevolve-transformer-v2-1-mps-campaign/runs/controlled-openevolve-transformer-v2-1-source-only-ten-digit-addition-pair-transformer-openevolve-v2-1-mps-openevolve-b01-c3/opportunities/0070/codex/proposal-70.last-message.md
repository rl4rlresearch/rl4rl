MECHANISM: Orthonormal LayerNorm-subspace projections

HYPOTHESIS: A 1,237-parameter model will retain at least 99% accuracy because both attention QKV and MLP input projections only receive mean-zero LayerNorm outputs; removing their 36 algebraically inactive input-direction parameters preserves their learned function classes.

INTENDED_EDIT: Replace dense QKV and MLP input maps with learned maps over an orthonormal seven-dimensional basis of the actual LayerNorm output subspace, while preserving the current 13 shared far-distance biases.

EVIDENCE: Further distance-bias sharing reduced the 1,271-parameter design to 94.79%, showing that head-specific positional biases are load-bearing; the current 1,273-parameter design reached 99.59%, motivating an orthogonal reduction that preserves those biases and removes exact projection redundancies instead.

<<<<<<< SEARCH
class MeanZeroEmbedding(nn.Embedding):
    """Embedding parameterized on the subspace orthogonal to all-ones."""

    def __init__(self, num_embeddings: int, embedding_dim: int):
        super().__init__(num_embeddings, embedding_dim - 1)
        self.register_buffer("basis", mean_zero_basis(embedding_dim), persistent=False)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        return super().forward(idx) @ self.basis.transpose(0, 1)


class VocabCenteredEmbedding(nn.Embedding):
=======
class MeanZeroEmbedding(nn.Embedding):
    """Embedding parameterized on the subspace orthogonal to all-ones."""

    def __init__(self, num_embeddings: int, embedding_dim: int):
        super().__init__(num_embeddings, embedding_dim - 1)
        self.register_buffer("basis", mean_zero_basis(embedding_dim), persistent=False)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        return super().forward(idx) @ self.basis.transpose(0, 1)


class MeanZeroInputLinear(nn.Linear):
    """Learned linear map defined directly on mean-zero inputs."""

    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__(in_features - 1, out_features, bias=bias)
        self.register_buffer("basis", mean_zero_basis(in_features), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x @ self.basis, self.weight, self.bias)


class VocabCenteredEmbedding(nn.Embedding):
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
=======
        self.qkv = MeanZeroInputLinear(d_model, 3 * d_model, bias=False)
>>>>>>> REPLACE

<<<<<<< SEARCH
class PairwiseTiedBiasLinear(nn.Linear):
    """Linear layer whose outputs form learned-bias quartets."""

    def __init__(self, in_features: int, out_features: int):
        if out_features % 4 != 0:
            raise ValueError("out_features must be divisible by four")
        super().__init__(in_features, out_features, bias=True)
        self.bias = None
        self.free_bias = nn.Parameter(torch.zeros(out_features // 4))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bias = self.free_bias.repeat_interleave(4)
        return F.linear(x, self.weight, bias)
=======
class PairwiseTiedBiasLinear(MeanZeroInputLinear):
    """Mean-zero-input linear layer whose outputs form bias quartets."""

    def __init__(self, in_features: int, out_features: int):
        if out_features % 4 != 0:
            raise ValueError("out_features must be divisible by four")
        super().__init__(in_features, out_features, bias=True)
        self.bias = None
        self.free_bias = nn.Parameter(torch.zeros(out_features // 4))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bias = self.free_bias.repeat_interleave(4)
        return F.linear(x @ self.basis, self.weight, bias)
>>>>>>> REPLACE