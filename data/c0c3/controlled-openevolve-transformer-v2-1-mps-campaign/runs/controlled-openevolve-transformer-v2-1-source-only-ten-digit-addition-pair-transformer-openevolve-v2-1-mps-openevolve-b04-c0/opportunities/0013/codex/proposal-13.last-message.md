MECHANISM: Single-position LayerNorm gauge fixing

HYPOTHESIS: Removing only the all-ones component of the final positional embedding will reduce the model from 1628 to 1627 parameters while retaining at least 99% accuracy, because this component is exactly erased by the pre-LayerNorm/final-LayerNorm architecture and affects only the causally last position’s optimizer coordinates.

INTENDED_EDIT: Store the final positional embedding in a seven-dimensional orthonormal zero-mean basis, preserve full-shaped random-number consumption and baseline-equivalent initialization, and leave all other positional embeddings unchanged.

EVIDENCE: The 1628-parameter model achieved 99.76%; quotienting all 23 positional embeddings reached only 50.86%, so isolating one exact gauge direction at the causally least influential position tests whether that failure resulted from perturbing every positional optimizer pathway simultaneously.

<<<<<<< SEARCH
@dataclass
class ModelConfig:
    n_layer: int
    d_model: int
    n_head: int
    d_ff: int
    dropout: float
    max_seq_len: int
    vocab_size: int


class CausalSelfAttention(nn.Module):
=======
@dataclass
class ModelConfig:
    n_layer: int
    d_model: int
    n_head: int
    d_ff: int
    dropout: float
    max_seq_len: int
    vocab_size: int


class LastPositionGaugeEmbedding(nn.Module):
    """Embedding with the final row represented modulo its unobservable mean."""

    def __init__(self, num_embeddings: int, embedding_dim: int):
        super().__init__()
        if num_embeddings < 1 or embedding_dim < 2:
            raise ValueError("gauge-fixed embedding requires at least one row and two dimensions")

        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.free_weight = nn.Parameter(torch.empty(num_embeddings - 1, embedding_dim))
        self.last_coord = nn.Parameter(torch.empty(embedding_dim - 1))

        basis = torch.zeros(embedding_dim, embedding_dim - 1)
        for j in range(embedding_dim - 1):
            scale = math.sqrt((j + 1) * (j + 2))
            basis[: j + 1, j] = 1.0 / scale
            basis[j + 1, j] = -(j + 1) / scale
        self.register_buffer("basis", basis, persistent=False)
        self.reset_parameters()

    @torch.no_grad()
    def reset_parameters(self) -> None:
        # Consume the same full-shaped draw as nn.Embedding, then discard only
        # the final row's functionally invisible all-ones component.
        virtual_weight = torch.empty(
            self.num_embeddings,
            self.embedding_dim,
            device=self.free_weight.device,
            dtype=self.free_weight.dtype,
        )
        nn.init.normal_(virtual_weight, mean=0.0, std=0.02)
        self.free_weight.copy_(virtual_weight[:-1])
        self.last_coord.copy_(virtual_weight[-1] @ self.basis)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        last_weight = self.last_coord @ self.basis.transpose(0, 1)
        weight = torch.cat((self.free_weight, last_weight.unsqueeze(0)), dim=0)
        return F.embedding(idx, weight)


class CausalSelfAttention(nn.Module):
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.token_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_emb = nn.Embedding(cfg.max_seq_len, cfg.d_model)
        self.drop = nn.Dropout(cfg.dropout)
=======
        self.token_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_emb = LastPositionGaugeEmbedding(cfg.max_seq_len, cfg.d_model)
        self.drop = nn.Dropout(cfg.dropout)
>>>>>>> REPLACE

<<<<<<< SEARCH
    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)
=======
    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, LastPositionGaugeEmbedding):
            module.reset_parameters()
>>>>>>> REPLACE