MECHANISM: Rank-seven learned token-logit factorization

HYPOTHESIS: Replacing the full-rank tied 114-by-8 token embedding with a learned rank-seven factorization will reduce the model from 1,626 to 1,568 parameters while retaining at least 99% accuracy, because it preserves seven adaptable lexical features and the entire proven attention/MLP computation.

INTENDED_EDIT: Factor the shared input/output embedding into learned vocabulary codes and a learned orthonormal-initialized feature basis, using the same factors for token lookup and logit projection.

EVIDENCE: The 1,626-parameter design reached 99.95%, while repeated one-parameter gauge extensions were brittle and the gated-MLP alternative reached only 70.49%; this preserves every load-bearing transformer-block parameter and instead challenges the untested assumption that the tied lexical interface needs eight independent dimensions.

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


class GaugeFixedPositionEmbedding(nn.Module):
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
    """Learned low-rank token embedding shared with the logit projection."""

    def __init__(self, num_embeddings: int, embedding_dim: int, rank: int):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.rank = rank
        self.code = nn.Parameter(torch.empty(num_embeddings, rank))
        self.basis = nn.Parameter(torch.empty(rank, embedding_dim))
        self.reset_parameters()

    @torch.no_grad()
    def reset_parameters(self, std: float = 1.0) -> None:
        nn.init.normal_(self.code, mean=0.0, std=std)
        nn.init.orthogonal_(self.basis)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        return F.embedding(idx, self.code) @ self.basis

    def logits(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x @ self.basis.t(), self.code)


class GaugeFixedPositionEmbedding(nn.Module):
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.cfg = cfg
        self.token_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_emb = GaugeFixedPositionEmbedding(cfg.max_seq_len, cfg.d_model)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.ln_f = nn.LayerNorm(cfg.d_model)

        # Weight tying with input embeddings.
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.token_emb.weight

        self.apply(self._init_weights)
=======
        self.cfg = cfg
        self.token_emb = FactorizedTokenEmbedding(
            cfg.vocab_size, cfg.d_model, cfg.d_model - 1
        )
        self.pos_emb = GaugeFixedPositionEmbedding(cfg.max_seq_len, cfg.d_model)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.ln_f = nn.LayerNorm(cfg.d_model)

        self.apply(self._init_weights)
>>>>>>> REPLACE

<<<<<<< SEARCH
    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, GaugeFixedPositionEmbedding):
            module.reset_parameters(std=0.02)
        elif isinstance(module, GaugeFixedBiasLinear):
=======
    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, FactorizedTokenEmbedding):
            module.reset_parameters(std=0.02)
        elif isinstance(module, GaugeFixedPositionEmbedding):
            module.reset_parameters(std=0.02)
        elif isinstance(module, GaugeFixedBiasLinear):
>>>>>>> REPLACE

<<<<<<< SEARCH
        x = self.ln_f(x)
        logits = self.lm_head(x)
=======
        x = self.ln_f(x)
        logits = self.token_emb.logits(x)
>>>>>>> REPLACE