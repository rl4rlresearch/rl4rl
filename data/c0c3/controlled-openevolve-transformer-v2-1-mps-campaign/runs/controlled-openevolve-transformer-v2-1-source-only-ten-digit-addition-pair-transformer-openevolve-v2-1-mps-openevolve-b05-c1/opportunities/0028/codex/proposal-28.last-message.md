MECHANISM: Exact quotienting of LayerNorm- and softmax-invariant parameter directions

HYPOTHESIS: Centering positional features, vocabulary embeddings, and residual-branch outputs removes 42 functionally redundant parameters, reducing the model from 1,384 to 1,342 parameters while retaining at least 99% accuracy after 60,000 steps.

INTENDED_EDIT: Replace full positional and token embeddings with learned mean-zero parameterizations, constrain attention and MLP residual outputs to the mean-zero feature subspace, preserve the essential one-neuron MLP, and extend training to 60,000 steps.

EVIDENCE: The 1,384-parameter one-neuron MLP model reached 99.99%, while removing that MLP collapsed accuracy to 73.76%; this motivates retaining its learned nonlinearity and instead removing only exact invariances induced by LayerNorm and softmax.

<<<<<<< SEARCH
class CausalSelfAttention(nn.Module):
=======
class MeanZeroOutputLinear(nn.Module):
    """Linear map whose outputs lie in the mean-zero feature subspace."""

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
        return self.linear(x) @ self.basis.T


class MeanZeroTokenEmbedding(nn.Module):
    """Embedding table centered across vocabulary rows."""

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
        self.table = nn.Embedding(num_embeddings - 1, embedding_dim)

    def full_weight(self) -> torch.Tensor:
        return self.basis @ self.table.weight

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        return F.embedding(idx, self.full_weight())


class MeanZeroFeatureEmbedding(nn.Module):
    """Embedding vectors restricted to the mean-zero feature subspace."""

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
        self.table = nn.Embedding(num_embeddings, embedding_dim - 1)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        return self.table(idx) @ self.basis.T


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
=======
        self.token_emb = MeanZeroTokenEmbedding(cfg.vocab_size, cfg.d_model)
        self.pos_emb = MeanZeroFeatureEmbedding(cfg.max_seq_len, cfg.d_model)
>>>>>>> REPLACE

<<<<<<< SEARCH
        # Weight tying with input embeddings.
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.token_emb.weight

        self.apply(self._init_weights)
=======
        self.apply(self._init_weights)
>>>>>>> REPLACE

<<<<<<< SEARCH
        logits = self.lm_head(x)
=======
        logits = F.linear(x, self.token_emb.full_weight())
>>>>>>> REPLACE

<<<<<<< SEARCH
    p.add_argument("--train-steps", type=int, default=52000)
=======
    p.add_argument("--train-steps", type=int, default=60000)
>>>>>>> REPLACE