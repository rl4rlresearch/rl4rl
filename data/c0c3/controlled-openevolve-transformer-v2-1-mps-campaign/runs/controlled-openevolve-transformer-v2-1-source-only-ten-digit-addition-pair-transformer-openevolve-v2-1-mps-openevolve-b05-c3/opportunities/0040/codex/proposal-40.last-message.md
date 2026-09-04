MECHANISM: Learned rank-seven tied token codebook

HYPOTHESIS: Replacing the full eight-dimensional tied token matrix with a learned rank-seven factorization will reduce the model from 1,644 to 1,586 parameters while retaining at least 99% accuracy, because the attention and MLP streams remain width eight and final LayerNorm already restricts centered output states to seven degrees of freedom.

INTENDED_EDIT: Encode every token through a learned seven-dimensional code and shared learned projection to width eight, then reuse the same factorization for output logits.

EVIDENCE: The width-six two-block design failed completely, showing that narrowing the entire computational stream is destructive, while the width-eight baseline reached 99.96%. This patch challenges the separate assumption that token identity and output classification require eight independent embedding dimensions without narrowing attention or MLP computation.

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


class FactorizedTokenEmbedding(nn.Module):
    """Learned rank-seven token representation shared with the output head."""

    def __init__(self, num_embeddings: int, embedding_dim: int):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.rank = embedding_dim - 1
        self.codes = nn.Embedding(num_embeddings, self.rank)
        self.proj = nn.Linear(self.rank, embedding_dim, bias=False)

    def full_weight(self) -> torch.Tensor:
        return self.codes.weight @ self.proj.weight.transpose(0, 1)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        return self.proj(self.codes(idx))


class CausalSelfAttention(nn.Module):
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
        self.token_emb = FactorizedTokenEmbedding(cfg.vocab_size, cfg.d_model)
        self.pos_emb = nn.Embedding(cfg.max_seq_len, cfg.d_model)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.ln_f = nn.LayerNorm(cfg.d_model)

        self.apply(self._init_weights)
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
        if isinstance(module, FactorizedTokenEmbedding):
            nn.init.normal_(
                module.codes.weight,
                mean=0.0,
                std=0.02 * math.sqrt(module.embedding_dim / module.rank),
            )
            nn.init.orthogonal_(module.proj.weight)
        elif isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)
>>>>>>> REPLACE

<<<<<<< SEARCH
        x = self.ln_f(x)
        logits = self.lm_head(x)
=======
        x = self.ln_f(x)
        logits = F.linear(x, self.token_emb.full_weight())
>>>>>>> REPLACE