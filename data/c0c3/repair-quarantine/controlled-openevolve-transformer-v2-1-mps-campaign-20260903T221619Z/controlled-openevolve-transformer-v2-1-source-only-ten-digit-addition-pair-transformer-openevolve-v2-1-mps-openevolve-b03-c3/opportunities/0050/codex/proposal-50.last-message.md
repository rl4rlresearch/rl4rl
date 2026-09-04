MECHANISM: Four-dimensional learned positional subspace

HYPOTHESIS: Replacing independent eight-dimensional position vectors with learned four-dimensional position codes and a shared learned basis will reduce parameters by 99 while retaining at least 99% accuracy; the qualified four-dimensional shared key/value stream suggests four positional routing coordinates are sufficient, whereas the failed two-dimensional routing design identifies a plausible lower boundary.

INTENDED_EDIT: Challenge the assumption that every position needs an independent full-width residual vector by factorizing the positional table into four-dimensional learned codes and a shared 4×8 basis, then train both factors with ordinary AdamW while preserving all qualified attention, MLP, gauge, and five-scale-factorization behavior.

EVIDENCE: The current four-dimensional shared key/value mechanism achieved 99.95% accuracy at 1,535 parameters, but reducing routing to two dimensions achieved only 81.88%; a four-dimensional positional subspace therefore tests a substantially smaller representation without repeating the demonstrated two-dimensional bottleneck.

<<<<<<< SEARCH
class GaugeFixedPositionEmbedding(nn.Module):
    """Position embedding with one shift-invariant scalar removed."""

    def __init__(self, num_embeddings: int, embedding_dim: int):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.first = nn.Parameter(torch.empty(embedding_dim - 1))
        self.rest = nn.Parameter(torch.empty(num_embeddings - 1, embedding_dim))
        self.full_first = None
        self.reset_parameters()

    @torch.no_grad()
    def reset_parameters(self, std: float = 1.0) -> None:
        raw = self.first.new_empty(self.num_embeddings, self.embedding_dim)
        nn.init.normal_(raw, mean=0.0, std=std)
        self.first.copy_(raw[0, :-1] - raw[0, -1])
        self.rest.copy_(raw[1:])

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        first = torch.cat((self.first, self.first.new_zeros(1)))
        if torch.is_grad_enabled():
            first.retain_grad()
            self.full_first = first
        weight = torch.cat((first.unsqueeze(0), self.rest), dim=0)
        return F.embedding(idx, weight)
=======
class LowRankPositionEmbedding(nn.Module):
    """Positions represented by learned codes in a shared latent subspace."""

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        rank: int = 4,
    ):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.rank = rank
        self.codes = nn.Parameter(torch.empty(num_embeddings, rank))
        self.basis = nn.Parameter(torch.empty(rank, embedding_dim))
        self.reset_parameters()

    @torch.no_grad()
    def reset_parameters(self, std: float = 1.0) -> None:
        nn.init.orthogonal_(self.basis)
        code_std = std * math.sqrt(self.embedding_dim / self.rank)
        nn.init.normal_(self.codes, mean=0.0, std=code_std)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        return F.embedding(idx, self.codes) @ self.basis
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.pos_emb = GaugeFixedPositionEmbedding(cfg.max_seq_len, cfg.d_model)
=======
        self.pos_emb = LowRankPositionEmbedding(
            cfg.max_seq_len, cfg.d_model, rank=4
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
        if isinstance(module, GaugeFixedPositionEmbedding):
            module.reset_parameters(std=0.02)
=======
        if isinstance(module, LowRankPositionEmbedding):
            module.reset_parameters(std=0.02)
>>>>>>> REPLACE

<<<<<<< SEARCH
    # Preserve full eight-coordinate AdamW dynamics for the positional,
    # attention-weight, attention-bias, terminal-bias, and four
    # terminal-weight gauges.
    gauge_params = [model.pos_emb.first]
=======
    # Preserve full eight-coordinate AdamW dynamics for the attention-weight,
    # attention-bias, terminal-bias, and four terminal-weight gauges.
    gauge_params = []
>>>>>>> REPLACE

<<<<<<< SEARCH
        full_gauge_grads = [model.pos_emb.full_first.grad.detach()]
=======
        full_gauge_grads = []
>>>>>>> REPLACE