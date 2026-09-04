MECHANISM: Shared rank-six token manifold

HYPOTHESIS: Replacing the full tied eight-dimensional token table with learned six-dimensional token codes and a shared learned projection will reduce the model from 1,384 to 1,204 parameters while retaining at least 99% accuracy after 56,000 steps.

INTENDED_EDIT: Factor both input embeddings and output logits through the same learned rank-six token representation, initialize its projection as a variance-preserving orthogonal map, and add 4,000 convergence steps.

EVIDENCE: The rank-six positional factorization reached 99.98% accuracy, showing that learned low-rank codes with a shared projection remain trainable; the current one-neuron model reached 99.99%, providing headroom to challenge the assumption that every vocabulary item requires eight independent coordinates.

<<<<<<< SEARCH
class TinyDecoderLM(nn.Module):
=======
class FactorizedTokenEmbedding(nn.Module):
    """Tied input/output embedding through a learned low-rank token manifold."""

    def __init__(self, num_embeddings: int, embedding_dim: int, rank: int):
        super().__init__()
        self.codes = nn.Embedding(num_embeddings, rank)
        self.proj = nn.Linear(rank, embedding_dim, bias=False)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        return self.proj(self.codes(idx))

    def logits(self, x: torch.Tensor) -> torch.Tensor:
        features = F.linear(x, self.proj.weight.transpose(0, 1))
        return F.linear(features, self.codes.weight)


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
        token_rank = min(6, cfg.d_model)
        self.token_emb = FactorizedTokenEmbedding(cfg.vocab_size, cfg.d_model, token_rank)
        self.pos_emb = nn.Embedding(cfg.max_seq_len, cfg.d_model)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.ln_f = nn.LayerNorm(cfg.d_model)

        self.apply(self._init_weights)
        nn.init.orthogonal_(
            self.token_emb.proj.weight,
            gain=math.sqrt(cfg.d_model / token_rank),
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
        x = self.ln_f(x)
        logits = self.lm_head(x)
=======
        x = self.ln_f(x)
        logits = self.token_emb.logits(x)
>>>>>>> REPLACE

<<<<<<< SEARCH
    p.add_argument("--train-steps", type=int, default=52000)
=======
    p.add_argument("--train-steps", type=int, default=56000)
>>>>>>> REPLACE