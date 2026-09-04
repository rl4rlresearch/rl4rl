MECHANISM: Full-width compositional position codebooks

HYPOTHESIS: Generic learned coarse/fine position codebooks will retain at least 99% accuracy while substantially reducing parameters because they preserve unique, full-width positional representations without requiring an independent vector per position.

INTENDED_EDIT: Replace the absolute position table with two learned full-width codebooks indexed compositionally, and train those parameters through ordinary AdamW instead of positional gauge bookkeeping.

EVIDENCE: The rank-4 learned positional subspace retained 97.06%, indicating positional compression is plausible but width-limited; the fixed full-width Chebyshev design reached only 94.19%, suggesting that learned discrete, non-smooth positional structure is more appropriate than a fixed polynomial span.

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
class FactorizedPositionEmbedding(nn.Module):
    """Full-width positions composed from learned coarse and fine codes."""

    def __init__(self, num_embeddings: int, embedding_dim: int):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.fine_size = math.ceil(math.sqrt(num_embeddings))
        self.coarse_size = math.ceil(num_embeddings / self.fine_size)
        self.coarse = nn.Parameter(
            torch.empty(self.coarse_size, embedding_dim)
        )
        self.fine = nn.Parameter(
            torch.empty(self.fine_size, embedding_dim)
        )
        self.reset_parameters()

    @torch.no_grad()
    def reset_parameters(self, std: float = 1.0) -> None:
        component_std = std / math.sqrt(2.0)
        nn.init.normal_(
            self.coarse, mean=0.0, std=component_std
        )
        nn.init.normal_(
            self.fine, mean=0.0, std=component_std
        )

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        coarse_idx = torch.div(
            idx, self.fine_size, rounding_mode="floor"
        )
        fine_idx = torch.remainder(idx, self.fine_size)
        return (
            F.embedding(coarse_idx, self.coarse)
            + F.embedding(fine_idx, self.fine)
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.pos_emb = GaugeFixedPositionEmbedding(cfg.max_seq_len, cfg.d_model)
=======
        self.pos_emb = FactorizedPositionEmbedding(
            cfg.max_seq_len, cfg.d_model
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
        elif isinstance(module, GaugeFixedPositionEmbedding):
            module.reset_parameters(std=0.02)
=======
        elif isinstance(module, FactorizedPositionEmbedding):
            module.reset_parameters(std=0.02)
>>>>>>> REPLACE

<<<<<<< SEARCH
    # Preserve full ambient-coordinate AdamW dynamics for the tied-token,
    # positional, three attention-weight, attention-bias, terminal-bias, and
    # six terminal-weight gauges.
    gauge_params = [model.token_emb.weight, model.pos_emb.first]
=======
    # Preserve full ambient-coordinate AdamW dynamics for the tied-token,
    # three attention-weight, attention-bias, terminal-bias, and six
    # terminal-weight gauges. The compositional position codes use AdamW
    # directly.
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