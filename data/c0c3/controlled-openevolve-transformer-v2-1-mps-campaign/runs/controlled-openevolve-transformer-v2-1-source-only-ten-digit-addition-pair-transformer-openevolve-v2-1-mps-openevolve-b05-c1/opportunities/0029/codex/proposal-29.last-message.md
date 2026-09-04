MECHANISM: Orthogonal quotient parameterization of exact LayerNorm and softmax invariances

HYPOTHESIS: Removing 42 redundant embedding and residual-output directions will reduce the model from 1,384 to 1,342 parameters while retaining at least 99% accuracy within the verified 52,000-step budget.

INTENDED_EDIT: Constrain vocabulary columns, positional rows, and both residual outputs to their invariant mean-zero subspaces, preserve tied input/output embeddings, and reconstruct full operators efficiently during forward passes.

EVIDENCE: The 1,384-parameter model reached 99.99% at 52,000 steps; the prior 1,342-parameter quotient attempt failed only because 60,000-step training exceeded the time limit, motivating the same exact quotient at the already successful budget with cheaper operator reconstruction.

<<<<<<< SEARCH
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x @ self.basis)


class CausalSelfAttention(nn.Module):
=======
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = self.linear.weight @ self.basis.T
        return F.linear(x, weight, self.linear.bias)


class MeanZeroOutputLinear(nn.Module):
    """Linear map restricted to the mean-zero output subspace."""

    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        if out_features < 2:
            raise ValueError("out_features must be at least two")

        basis = torch.zeros(out_features, out_features - 1)
        for j in range(out_features - 1):
            scale = math.sqrt((j + 1) * (j + 2))
            basis[: j + 1, j] = 1.0 / scale
            basis[j + 1, j] = -(j + 1) / scale
        self.register_buffer("basis", basis, persistent=False)
        self.linear = nn.Linear(in_features, out_features - 1, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = self.basis @ self.linear.weight
        bias = None if self.linear.bias is None else self.basis @ self.linear.bias
        return F.linear(x, weight, bias)


class MeanZeroFeatureEmbedding(nn.Module):
    """Embedding whose feature vectors have zero mean."""

    def __init__(self, num_embeddings: int, embedding_dim: int):
        super().__init__()
        if embedding_dim < 2:
            raise ValueError("embedding_dim must be at least two")

        basis = torch.zeros(embedding_dim, embedding_dim - 1)
        for j in range(embedding_dim - 1):
            scale = math.sqrt((j + 1) * (j + 2))
            basis[: j + 1, j] = 1.0 / scale
            basis[j + 1, j] = -(j + 1) / scale
        self.register_buffer("basis", basis, persistent=False)
        self.embedding = nn.Embedding(num_embeddings, embedding_dim - 1)

    def full_weight(self) -> torch.Tensor:
        return self.embedding.weight @ self.basis.T

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        return F.embedding(idx, self.full_weight())


class MeanZeroVocabEmbedding(nn.Module):
    """Tied vocabulary table centered across token types."""

    def __init__(self, num_embeddings: int, embedding_dim: int):
        super().__init__()
        if num_embeddings < 2:
            raise ValueError("num_embeddings must be at least two")

        basis = torch.zeros(num_embeddings, num_embeddings - 1)
        for j in range(num_embeddings - 1):
            scale = math.sqrt((j + 1) * (j + 2))
            basis[: j + 1, j] = 1.0 / scale
            basis[j + 1, j] = -(j + 1) / scale
        self.register_buffer("basis", basis, persistent=False)
        self.embedding = nn.Embedding(num_embeddings - 1, embedding_dim)

    def full_weight(self) -> torch.Tensor:
        return self.basis @ self.embedding.weight

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        return F.embedding(idx, self.full_weight())

    def project(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.full_weight())


class CausalSelfAttention(nn.Module):
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.proj = nn.Linear(d_model, d_model)
=======
        self.proj = MeanZeroOutputLinear(d_model, d_model)
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.fc2 = nn.Linear(d_ff, d_model)
=======
        self.fc2 = MeanZeroOutputLinear(d_ff, d_model)
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.token_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_emb = nn.Embedding(cfg.max_seq_len, cfg.d_model)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.ln_f = nn.LayerNorm(cfg.d_model)

        # Weight tying with input embeddings.
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.token_emb.weight
=======
        self.token_emb = MeanZeroVocabEmbedding(cfg.vocab_size, cfg.d_model)
        self.pos_emb = MeanZeroFeatureEmbedding(cfg.max_seq_len, cfg.d_model)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.ln_f = nn.LayerNorm(cfg.d_model)
>>>>>>> REPLACE

<<<<<<< SEARCH
        logits = self.lm_head(x)
=======
        logits = self.token_emb.project(x)
>>>>>>> REPLACE