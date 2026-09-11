MECHANISM: Global tied-embedding common-mode anchoring

HYPOTHESIS: Removing the single global all-ones component of the tied token/output embedding will reduce the qualified model from 1583 to 1582 parameters while retaining at least 99% accuracy, because it only shifts residual streams by a LayerNorm-invisible constant and shifts every output logit equally.

INTENDED_EDIT: Replace the tied embedding with a gauge-fixed parameterization in which the final token vector is mean-zero, using the exact global shift symmetry to remove one scalar while preserving initialization and weight tying.

EVIDENCE: The current 1583-parameter model achieved 99.96%. Full eight-dimensional vocabulary centering failed at 71.46%, motivating this narrower one-dimensional gauge that requires no positional-embedding compensation and leaves all other token coordinates directly parameterized.

<<<<<<< SEARCH
    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        reduced = F.embedding(idx, self.weight)
        return reduced @ self.basis.transpose(0, 1)


class MeanZeroOutputLinear(nn.Linear):
=======
    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        reduced = F.embedding(idx, self.weight)
        return reduced @ self.basis.transpose(0, 1)


class GaugeFixedEmbedding(nn.Module):
    """Tied embedding with its global all-ones gauge fixed."""
    def __init__(self, num_embeddings: int, embedding_dim: int):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        basis = _mean_zero_basis(embedding_dim)
        self.register_buffer("basis", basis, persistent=False)

        full_weight = torch.empty(num_embeddings, embedding_dim)
        nn.init.normal_(full_weight)
        anchor = full_weight[-1].mean()
        gauged_weight = full_weight - anchor
        self.weight_rows = nn.Parameter(gauged_weight[:-1])
        self.last_weight = nn.Parameter(gauged_weight[-1] @ basis)

    def reset_from_full(self, full_weight: torch.Tensor) -> None:
        anchor = full_weight[-1].mean()
        gauged_weight = full_weight - anchor
        with torch.no_grad():
            self.weight_rows.copy_(gauged_weight[:-1])
            self.last_weight.copy_(gauged_weight[-1] @ self.basis)

    @property
    def weight(self) -> torch.Tensor:
        last_row = self.basis @ self.last_weight
        return torch.cat((self.weight_rows, last_row.unsqueeze(0)), dim=0)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        return F.embedding(idx, self.weight)


class TiedOutputLinear(nn.Module):
    """Parameter-free output view of a gauge-fixed input embedding."""
    def __init__(self, embedding: GaugeFixedEmbedding):
        super().__init__()
        object.__setattr__(self, "embedding", embedding)

        # Match the discarded initialization performed by a tied nn.Linear.
        discarded_weight = torch.empty(
            embedding.num_embeddings, embedding.embedding_dim
        )
        nn.init.kaiming_uniform_(discarded_weight, a=math.sqrt(5))

    @property
    def weight(self) -> torch.Tensor:
        return self.embedding.weight

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight)


class MeanZeroOutputLinear(nn.Linear):
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.cfg = cfg
        self.token_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_emb = MeanZeroEmbedding(cfg.max_seq_len, cfg.d_model)
=======
        self.cfg = cfg
        self.token_emb = GaugeFixedEmbedding(cfg.vocab_size, cfg.d_model)
        self.pos_emb = MeanZeroEmbedding(cfg.max_seq_len, cfg.d_model)
>>>>>>> REPLACE

<<<<<<< SEARCH
        # Weight tying with input embeddings.
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.token_emb.weight
=======
        # Weight tying with input embeddings.
        self.lm_head = TiedOutputLinear(self.token_emb)
>>>>>>> REPLACE

<<<<<<< SEARCH
    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, MeanZeroEmbedding):
=======
    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, GaugeFixedEmbedding):
            full_weight = module.weight_rows.new_empty(
                module.num_embeddings, module.embedding_dim
            )
            nn.init.normal_(full_weight, mean=0.0, std=0.02)
            module.reset_from_full(full_weight)
        elif isinstance(module, TiedOutputLinear):
            embedding = module.embedding
            full_weight = embedding.weight_rows.new_empty(
                embedding.num_embeddings, embedding.embedding_dim
            )
            nn.init.normal_(full_weight, mean=0.0, std=0.02)
            embedding.reset_from_full(full_weight)
        elif isinstance(module, MeanZeroEmbedding):
>>>>>>> REPLACE