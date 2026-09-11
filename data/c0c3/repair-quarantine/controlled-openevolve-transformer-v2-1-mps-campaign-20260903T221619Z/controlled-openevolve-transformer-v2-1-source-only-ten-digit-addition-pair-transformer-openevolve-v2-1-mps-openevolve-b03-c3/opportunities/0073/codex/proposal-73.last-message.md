MECHANISM: Second independent positional scalar-shift gauge

HYPOTHESIS: Removing a second positional-row scalar will produce a 1,524-parameter model with at least 99% accuracy because this exact pre-LayerNorm shift symmetry already succeeds for the first positional row.

INTENDED_EDIT: Gauge-fix the second positional embedding row and include its omitted ambient coordinate in AdamW moments, gradient clipping, decay, and updates.

EVIDENCE: The current 1,525-parameter design reached 99.93% with one positional scalar gauge, while the third ln1 absorption fell to 98.85%; this tests a distinct exact symmetry with a one-parameter reduction.

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
=======
class GaugeFixedPositionEmbedding(nn.Module):
    """Embedding with two independent positional scalar shifts removed."""

    def __init__(self, num_embeddings: int, embedding_dim: int):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.first = nn.Parameter(torch.empty(embedding_dim - 1))
        self.second = nn.Parameter(torch.empty(embedding_dim - 1))
        self.rest = nn.Parameter(torch.empty(num_embeddings - 2, embedding_dim))
        self.full_first = None
        self.full_second = None
        self.reset_parameters()
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.first.copy_(raw[0, :-1] - raw[0, -1])
        self.rest.copy_(raw[1:])
=======
        self.first.copy_(raw[0, :-1] - raw[0, -1])
        self.second.copy_(raw[1, :-1] - raw[1, -1])
        self.rest.copy_(raw[2:])
>>>>>>> REPLACE

<<<<<<< SEARCH
    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        first = torch.cat((self.first, self.first.new_zeros(1)))
        if torch.is_grad_enabled():
            first.retain_grad()
            self.full_first = first
        weight = torch.cat((first.unsqueeze(0), self.rest), dim=0)
        return F.embedding(idx, weight)
=======
    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        first = torch.cat((self.first, self.first.new_zeros(1)))
        second = torch.cat((self.second, self.second.new_zeros(1)))
        if torch.is_grad_enabled():
            first.retain_grad()
            second.retain_grad()
            self.full_first = first
            self.full_second = second
        weight = torch.cat(
            (first.unsqueeze(0), second.unsqueeze(0), self.rest),
            dim=0,
        )
        return F.embedding(idx, weight)
>>>>>>> REPLACE

<<<<<<< SEARCH
    # Preserve full ambient-coordinate AdamW dynamics for the tied-token,
    # positional, three attention-weight, attention-bias, terminal-bias, and
    # six terminal-weight gauges.
    gauge_params = [model.token_emb.weight, model.pos_emb.first]
=======
    # Preserve full ambient-coordinate AdamW dynamics for the tied-token,
    # two positional, three attention-weight, attention-bias, terminal-bias,
    # and six terminal-weight gauges.
    gauge_params = [
        model.token_emb.weight,
        model.pos_emb.first,
        model.pos_emb.second,
    ]
>>>>>>> REPLACE

<<<<<<< SEARCH
        full_gauge_grads = [
            model.token_emb.full_weight.grad.detach().reshape(-1),
            model.pos_emb.full_first.grad.detach(),
        ]
=======
        full_gauge_grads = [
            model.token_emb.full_weight.grad.detach().reshape(-1),
            model.pos_emb.full_first.grad.detach(),
            model.pos_emb.full_second.grad.detach(),
        ]
>>>>>>> REPLACE