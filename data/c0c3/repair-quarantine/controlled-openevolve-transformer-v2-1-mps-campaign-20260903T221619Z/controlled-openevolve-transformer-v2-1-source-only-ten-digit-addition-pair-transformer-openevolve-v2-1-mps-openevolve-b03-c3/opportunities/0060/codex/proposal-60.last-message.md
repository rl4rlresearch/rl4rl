MECHANISM: Full-width orthogonal-polynomial positional representation

HYPOTHESIS: A 1,334-parameter model will retain at least 99% accuracy because a fixed eight-dimensional Chebyshev position basis preserves full residual width and lets learned projections express structured forward or reversed positional relationships, avoiding the four-dimensional bottleneck that limited the learned positional-subspace design to 97.06%.

INTENDED_EDIT: Challenge the assumption that every sequence position needs an independent learned vector; replace the gauge-fixed positional table with generic fixed polynomial features and a learned 8×8 mixer, while training that mixer through ordinary AdamW and retaining the qualified attention, MLP, and gauge machinery.

EVIDENCE: The learned four-dimensional positional subspace reached 97.06%, showing that structured positional compression retained most task behavior but that four routing dimensions were insufficient; this patch restores all eight positional dimensions while using only 64 learned positional parameters instead of an independent full-width table.

<<<<<<< SEARCH
class GaugeFixedPositionEmbedding(nn.Module):
    """Embedding with one shift-invariant positional scalar removed."""

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
class PolynomialPositionEmbedding(nn.Module):
    """Full-width learned mixing of a fixed generic position basis."""

    def __init__(self, num_embeddings: int, embedding_dim: int):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.mix = nn.Parameter(
            torch.empty(embedding_dim, embedding_dim)
        )

        position = torch.linspace(-1.0, 1.0, num_embeddings)
        columns = [torch.ones_like(position)]
        if embedding_dim > 1:
            columns.append(position)
        for _ in range(2, embedding_dim):
            columns.append(
                2.0 * position * columns[-1] - columns[-2]
            )
        features = torch.stack(columns, dim=1) / math.sqrt(embedding_dim)
        self.register_buffer("features", features, persistent=False)
        self.reset_parameters()

    @torch.no_grad()
    def reset_parameters(self, std: float = 1.0) -> None:
        # Consume the original full-table draw so all other initialization
        # remains controlled while this mixer starts with the same marginal.
        raw = self.mix.new_empty(
            self.num_embeddings, self.embedding_dim
        )
        nn.init.normal_(raw, mean=0.0, std=std)
        self.mix.copy_(raw[: self.embedding_dim])

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        features = F.embedding(idx, self.features)
        return features @ self.mix
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.pos_emb = GaugeFixedPositionEmbedding(cfg.max_seq_len, cfg.d_model)
=======
        self.pos_emb = PolynomialPositionEmbedding(
            cfg.max_seq_len, cfg.d_model
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
        elif isinstance(module, GaugeFixedPositionEmbedding):
            module.reset_parameters(std=0.02)
=======
        elif isinstance(module, PolynomialPositionEmbedding):
            module.reset_parameters(std=0.02)
>>>>>>> REPLACE

<<<<<<< SEARCH
    # Preserve full ambient-coordinate AdamW dynamics for the tied-token,
    # positional, attention-weight, attention-bias, terminal-bias, and five
    # terminal-weight gauges.
    gauge_params = [model.token_emb.weight, model.pos_emb.first]
=======
    # Preserve full ambient-coordinate AdamW dynamics for the tied-token,
    # attention-weight, attention-bias, terminal-bias, and five terminal-weight
    # gauges. The learned polynomial mixer uses ordinary AdamW coordinates.
    gauge_params = [model.token_emb.weight]
>>>>>>> REPLACE

<<<<<<< SEARCH
        full_gauge_grads = [
            model.token_emb.full_weight.grad.detach().reshape(-1),
            model.pos_emb.full_first.grad.detach(),
        ]
=======
        full_gauge_grads = [
            model.token_emb.full_weight.grad.detach().reshape(-1),
        ]
>>>>>>> REPLACE