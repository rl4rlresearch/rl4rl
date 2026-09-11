MECHANISM: Optimizer-aligned token-row-mean quotient

HYPOTHESIS: Orthogonally isolating the decay-free token-row-mean output-bias coordinates before removing a seventh final-LayerNorm bias direction will produce a 1,579-parameter model with at least 99% accuracy.

INTENDED_EDIT: Reparameterize the unchanged globally mean-free tied embedding into row-centered content and explicit mean-free row offsets, keep both components free of weight decay, and reduce the final-LayerNorm learned bias basis from two coordinates to one.

EVIDENCE: Removing embedding weight decay raised the four-direction quotient from 98.28% to 99.95%, and the current six-direction quotient still reaches 99.13%; this indicates that optimization of the token-row-mean replacement channel is decisive and motivates aligning that channel with independent Adam coordinates before pruning another bias direction.

<<<<<<< SEARCH
class MeanFreeTokenEmbedding(nn.Embedding):
    """Tied embedding modulo its globally constant, functionally null shift."""

    def __init__(self, num_embeddings: int, embedding_dim: int):
        super().__init__(num_embeddings, embedding_dim)
        full_weight = self.weight.detach()
        size = num_embeddings * embedding_dim

        basis = torch.zeros(size, size - 1)
        for j in range(size - 1):
            scale = math.sqrt((j + 1) * (j + 2))
            basis[: j + 1, j] = 1.0 / scale
            basis[j + 1, j] = -(j + 1) / scale
        self.register_buffer("basis", basis, persistent=False)

        self.weight = nn.Parameter(
            (full_weight.reshape(-1) @ basis).clone()
        )

    def full_weight(self) -> torch.Tensor:
        return (self.basis @ self.weight).view(
            self.num_embeddings, self.embedding_dim
        )

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        return F.embedding(idx, self.full_weight())
=======
class MeanFreeTokenEmbedding(nn.Embedding):
    """Globally mean-free tied embedding with isolated token-row means."""

    def __init__(self, num_embeddings: int, embedding_dim: int):
        super().__init__(num_embeddings, embedding_dim)
        full_weight = self.weight.detach()

        content_basis = torch.zeros(embedding_dim, embedding_dim - 1)
        for j in range(embedding_dim - 1):
            scale = math.sqrt((j + 1) * (j + 2))
            content_basis[: j + 1, j] = 1.0 / scale
            content_basis[j + 1, j] = -(j + 1) / scale
        self.register_buffer("content_basis", content_basis, persistent=False)

        mean_basis = torch.zeros(num_embeddings, num_embeddings - 1)
        for j in range(num_embeddings - 1):
            scale = math.sqrt((j + 1) * (j + 2))
            mean_basis[: j + 1, j] = 1.0 / scale
            mean_basis[j + 1, j] = -(j + 1) / scale
        self.register_buffer("mean_basis", mean_basis, persistent=False)

        row_average = full_weight.mean(dim=1)
        self.weight = nn.Parameter(
            ((full_weight - row_average.unsqueeze(1)) @ content_basis).clone()
        )
        self.row_mean = nn.Parameter(
            (math.sqrt(embedding_dim) * (row_average @ mean_basis)).clone()
        )

    def full_weight(self) -> torch.Tensor:
        centered = self.weight @ self.content_basis.transpose(0, 1)
        row_offsets = (
            (self.mean_basis @ self.row_mean).unsqueeze(1)
            / math.sqrt(self.embedding_dim)
        )
        return centered + row_offsets

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        return F.embedding(idx, self.full_weight())
>>>>>>> REPLACE

<<<<<<< SEARCH
class TiedMeanFreeOutput(nn.Linear):
    """Output projection sharing the quotient embedding's sole parameter."""
=======
class TiedMeanFreeOutput(nn.Linear):
    """Output projection sharing the quotient embedding's parameters."""
>>>>>>> REPLACE

<<<<<<< SEARCH
                full.sub_(full.mean(dim=1, keepdim=True))
                embedding.weight.copy_(full.reshape(-1) @ embedding.basis)
=======
                full.sub_(full.mean(dim=1, keepdim=True))
                embedding.weight.copy_(full @ embedding.content_basis)
                embedding.row_mean.zero_()
>>>>>>> REPLACE

<<<<<<< SEARCH
                full.sub_(full.mean(dim=1, keepdim=True))
                module.weight.copy_(full.reshape(-1) @ module.basis)
=======
                full.sub_(full.mean(dim=1, keepdim=True))
                module.weight.copy_(full @ module.content_basis)
                module.row_mean.zero_()
>>>>>>> REPLACE

<<<<<<< SEARCH
class QuotientFinalLayerNorm(nn.LayerNorm):
    """Final LayerNorm with six bias directions absorbed by token-row means."""

    def __init__(self, normalized_shape: int):
        super().__init__(normalized_shape)

        basis = torch.zeros(normalized_shape, normalized_shape - 6)
        for j in range(normalized_shape - 6):
            scale = math.sqrt((j + 1) * (j + 2))
            basis[: j + 1, j] = 1.0 / scale
            basis[j + 1, j] = -(j + 1) / scale
        self.register_buffer("bias_basis", basis, persistent=False)
        self.register_buffer(
            "fixed_bias",
            torch.full((normalized_shape,), 1.0 / math.sqrt(normalized_shape)),
            persistent=False,
        )
        self.bias = nn.Parameter(torch.zeros(normalized_shape - 6))
=======
class QuotientFinalLayerNorm(nn.LayerNorm):
    """Final LayerNorm with seven bias directions absorbed by token-row means."""

    def __init__(self, normalized_shape: int):
        super().__init__(normalized_shape)

        basis = torch.zeros(normalized_shape, normalized_shape - 7)
        for j in range(normalized_shape - 7):
            scale = math.sqrt((j + 1) * (j + 2))
            basis[: j + 1, j] = 1.0 / scale
            basis[j + 1, j] = -(j + 1) / scale
        self.register_buffer("bias_basis", basis, persistent=False)
        self.register_buffer(
            "fixed_bias",
            torch.full((normalized_shape,), 1.0 / math.sqrt(normalized_shape)),
            persistent=False,
        )
        self.bias = nn.Parameter(torch.zeros(normalized_shape - 7))
>>>>>>> REPLACE

<<<<<<< SEARCH
    embedding_param = model.token_emb.weight
    decay_params = [
        param for param in model.parameters() if param is not embedding_param
    ]
    optimizer = torch.optim.AdamW(
        [
            {"params": decay_params, "weight_decay": train_cfg.weight_decay},
            {"params": [embedding_param], "weight_decay": 0.0},
        ],
        lr=train_cfg.lr,
    )
=======
    embedding_params = [model.token_emb.weight, model.token_emb.row_mean]
    decay_params = [
        param
        for param in model.parameters()
        if param is not model.token_emb.weight
        and param is not model.token_emb.row_mean
    ]
    optimizer = torch.optim.AdamW(
        [
            {"params": decay_params, "weight_decay": train_cfg.weight_decay},
            {"params": embedding_params, "weight_decay": 0.0},
        ],
        lr=train_cfg.lr,
    )
>>>>>>> REPLACE