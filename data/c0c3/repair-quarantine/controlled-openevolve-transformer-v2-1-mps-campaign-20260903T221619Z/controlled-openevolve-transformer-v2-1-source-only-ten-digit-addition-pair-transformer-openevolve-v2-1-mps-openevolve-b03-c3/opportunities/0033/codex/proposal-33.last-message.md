MECHANISM: Second-row positional shift gauge with ambient AdamW

HYPOTHESIS: Extending the qualified positional gauge to the second position row will reduce the verified 1,550-parameter design to 1,549 parameters while retaining at least 99% accuracy, because an independent all-ones shift at that position is erased by each pre-norm LayerNorm and the final LayerNorm, while full eight-coordinate AdamW dynamics are preserved.

INTENDED_EDIT: Reproduce the verified four-column terminal gauge and bias-free shared value projection, then gauge-fix a second positional row and include it in ambient-coordinate optimization.

EVIDENCE: The shared-key/value, four-terminal-gauge, bias-free-value design achieved 99.85% accuracy at 1,550 parameters; the existing first-row positional gauge establishes the same exact invariance and optimizer treatment for an adjacent positional row.

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
    """Embedding with two shift-invariant positional scalars removed."""

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

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        first = torch.cat((self.first, self.first.new_zeros(1)))
        if torch.is_grad_enabled():
            first.retain_grad()
            self.full_first = first
        weight = torch.cat((first.unsqueeze(0), self.rest), dim=0)
        return F.embedding(idx, weight)
=======
        self.first.copy_(raw[0, :-1] - raw[0, -1])
        self.second.copy_(raw[1, :-1] - raw[1, -1])
        self.rest.copy_(raw[2:])

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        first = torch.cat((self.first, self.first.new_zeros(1)))
        second = torch.cat((self.second, self.second.new_zeros(1)))
        if torch.is_grad_enabled():
            first.retain_grad()
            second.retain_grad()
            self.full_first = first
            self.full_second = second
        weight = torch.cat(
            (first.unsqueeze(0), second.unsqueeze(0), self.rest), dim=0
        )
        return F.embedding(idx, weight)
>>>>>>> REPLACE

<<<<<<< SEARCH
class GaugeFixedTerminalLinear(nn.Module):
    """Linear layer with bias and three weight-column output gauges removed."""
=======
class GaugeFixedTerminalLinear(nn.Module):
    """Linear layer with bias and four weight-column output gauges removed."""
>>>>>>> REPLACE

<<<<<<< SEARCH
                nn.Parameter(torch.empty(out_features - 1))
                for _ in range(3)
            ]
        )
        self.weight_rest = nn.Parameter(
            torch.empty(out_features, in_features - 3)
        )
=======
                nn.Parameter(torch.empty(out_features - 1))
                for _ in range(4)
            ]
        )
        self.weight_rest = nn.Parameter(
            torch.empty(out_features, in_features - 4)
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.weight_rest.copy_(raw_weight[:, 3:])
        self.bias.copy_(raw_bias[:-1] - raw_bias[-1])
=======
        self.weight_rest.copy_(raw_weight[:, 4:])
        self.bias.copy_(raw_bias[:-1] - raw_bias[-1])
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.q_bias = nn.Parameter(torch.zeros(d_model))
        self.v_bias = nn.Parameter(torch.zeros(self.head_dim))
        self.proj = nn.Linear(d_model, d_model)
=======
        self.q_bias = nn.Parameter(torch.zeros(d_model))
        self.proj = nn.Linear(d_model, d_model)
>>>>>>> REPLACE

<<<<<<< SEARCH
        q = self.q_proj(x) + self.q_bias
        k = self.k_proj(x)
        v = self.v_proj(x) + self.v_bias
=======
        q = self.q_proj(x) + self.q_bias
        k = self.k_proj(x)
        v = self.v_proj(x)
>>>>>>> REPLACE

<<<<<<< SEARCH
                module.weight_rest.copy_(raw_weight[:, 3:])
                nn.init.zeros_(module.bias)
=======
                module.weight_rest.copy_(raw_weight[:, 4:])
                nn.init.zeros_(module.bias)
>>>>>>> REPLACE

<<<<<<< SEARCH
    # Preserve full eight-coordinate AdamW dynamics for the positional,
    # terminal-bias, and three terminal-weight gauge vectors.
    gauge_params = [model.pos_emb.first]
=======
    # Preserve full eight-coordinate AdamW dynamics for both positional,
    # terminal-bias, and four terminal-weight gauge vectors.
    gauge_params = [model.pos_emb.first, model.pos_emb.second]
>>>>>>> REPLACE

<<<<<<< SEARCH
        full_gauge_grads = [model.pos_emb.full_first.grad.detach()]
        for blk in model.blocks:
=======
        full_gauge_grads = [
            model.pos_emb.full_first.grad.detach(),
            model.pos_emb.full_second.grad.detach(),
        ]
        for blk in model.blocks:
>>>>>>> REPLACE