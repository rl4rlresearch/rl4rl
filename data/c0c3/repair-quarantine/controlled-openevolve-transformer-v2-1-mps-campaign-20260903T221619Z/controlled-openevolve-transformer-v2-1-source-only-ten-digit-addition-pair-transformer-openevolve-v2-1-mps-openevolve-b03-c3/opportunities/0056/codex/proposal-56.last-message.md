MECHANISM: Global tied-token-embedding shift gauge with ambient AdamW

HYPOTHESIS: Removing the tied embedding’s single global scalar-shift degree of freedom will produce a 1,534-parameter model with at least 99% accuracy because LayerNorm erases its input-side effect, softmax erases its common-logit output effect, and ambient-coordinate AdamW preserves the qualified optimization dynamics.

INTENDED_EDIT: Replace the full tied token table with an exactly gauge-fixed table containing one fewer learned scalar, retain the generic learned output head, preserve initialization RNG order, and extend the existing ambient moments, clipping, and updates to the omitted embedding coordinate.

EVIDENCE: The current five-scale design achieved 99.95% accuracy at 1,535 parameters while already succeeding with an ambient additive positional gauge; the second positional-row gauge reached only 98.60%, motivating this distinct global symmetry whose input and output effects are both exactly unobservable.

<<<<<<< SEARCH
class GaugeFixedPositionEmbedding(nn.Module):
    """Position embedding with one shift-invariant scalar removed."""
=======
class GaugeFixedTokenEmbedding(nn.Module):
    """Tied embedding with its global scalar-shift gauge removed."""

    def __init__(self, num_embeddings: int, embedding_dim: int):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.weight = nn.Parameter(
            torch.empty(num_embeddings * embedding_dim - 1)
        )
        self.full_weight = None
        self.reset_parameters()

    @torch.no_grad()
    def reset_parameters(self, std: float = 1.0) -> None:
        raw = self.weight.new_empty(
            self.num_embeddings, self.embedding_dim
        )
        nn.init.normal_(raw, mean=0.0, std=std)
        flat = raw.reshape(-1)
        self.weight.copy_(flat[:-1] - flat[-1])

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        full_weight = torch.cat(
            (self.weight, self.weight.new_zeros(1))
        ).view(self.num_embeddings, self.embedding_dim)
        if torch.is_grad_enabled():
            full_weight.retain_grad()
        self.full_weight = full_weight
        return F.embedding(idx, full_weight)

    def project(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.full_weight)


class TiedTokenProjection(nn.Module):
    """Parameter-free output view of the learned token embedding."""

    def __init__(self, embedding: GaugeFixedTokenEmbedding):
        super().__init__()
        self.in_features = embedding.embedding_dim
        self.out_features = embedding.num_embeddings
        object.__setattr__(self, "embedding", embedding)

        # Match the constructor RNG consumption of the replaced nn.Linear.
        scratch = torch.empty(self.out_features, self.in_features)
        nn.init.kaiming_uniform_(scratch, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.embedding.project(x)


class GaugeFixedPositionEmbedding(nn.Module):
    """Position embedding with one shift-invariant scalar removed."""
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.token_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_emb = GaugeFixedPositionEmbedding(cfg.max_seq_len, cfg.d_model)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.ln_f = nn.LayerNorm(cfg.d_model)

        # Weight tying with input embeddings.
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.token_emb.weight
=======
        self.token_emb = GaugeFixedTokenEmbedding(
            cfg.vocab_size, cfg.d_model
        )
        self.pos_emb = GaugeFixedPositionEmbedding(cfg.max_seq_len, cfg.d_model)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.ln_f = nn.LayerNorm(cfg.d_model)

        # Parameter-free output view preserves input/output weight tying.
        self.lm_head = TiedTokenProjection(self.token_emb)
>>>>>>> REPLACE

<<<<<<< SEARCH
    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, GaugeFixedPositionEmbedding):
            module.reset_parameters(std=0.02)
        elif isinstance(module, GaugeFixedAttentionProjection):
=======
    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, GaugeFixedTokenEmbedding):
            module.reset_parameters(std=0.02)
        elif isinstance(module, TiedTokenProjection):
            # Match the second initialization of the formerly tied Linear.
            module.embedding.reset_parameters(std=0.02)
        elif isinstance(module, GaugeFixedPositionEmbedding):
            module.reset_parameters(std=0.02)
        elif isinstance(module, GaugeFixedAttentionProjection):
>>>>>>> REPLACE

<<<<<<< SEARCH
    # Preserve full eight-coordinate AdamW dynamics for the positional,
    # attention-weight, attention-bias, terminal-bias, and four
    # terminal-weight gauges.
    gauge_params = [model.pos_emb.first]
=======
    # Preserve full ambient-coordinate AdamW dynamics for the tied-token,
    # positional, attention-weight, attention-bias, terminal-bias, and four
    # terminal-weight gauges.
    gauge_params = [model.token_emb.weight, model.pos_emb.first]
>>>>>>> REPLACE

<<<<<<< SEARCH
        full_gauge_grads = [model.pos_emb.full_first.grad.detach()]
=======
        full_gauge_grads = [
            model.token_emb.full_weight.grad.detach().reshape(-1),
            model.pos_emb.full_first.grad.detach(),
        ]
>>>>>>> REPLACE