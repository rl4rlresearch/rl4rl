MECHANISM: Global tied-embedding shift gauge

HYPOTHESIS: Gauge-fixing the single global all-ones shift of the verified 1,542-parameter tied embedding will yield 1,541 parameters while retaining at least 99% accuracy, because LayerNorm removes the shift throughout the input path and the tied output path changes every vocabulary logit by the same scalar.

INTENDED_EDIT: Reproduce the qualified four-column, bias-free-value, bias-free-`ln1` design, then store the tied embedding relative to one fixed scalar and train it with full ambient-coordinate AdamW moments and clipping.

EVIDENCE: The qualified 1,542-parameter design achieved 99.91% accuracy. Its existing positional and terminal gauges show that exact shift invariances remain trainable when omitted coordinates retain ambient AdamW dynamics, motivating a distinct one-scalar gauge instead of extending the positional gauge that failed at 1,549 parameters.

<<<<<<< SEARCH
class GaugeFixedPositionEmbedding(nn.Module):
=======
class GaugeFixedTiedEmbedding(nn.Module):
    """Tied embedding with its global all-ones shift gauge removed."""

    def __init__(self, num_embeddings: int, embedding_dim: int):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.weight_prefix = nn.Parameter(
            torch.empty(num_embeddings * embedding_dim - 1)
        )
        self.full_weight = None
        self.reset_parameters()

    @torch.no_grad()
    def reset_parameters(self, std: float = 1.0) -> None:
        raw = self.weight_prefix.new_empty(
            self.num_embeddings, self.embedding_dim
        )
        nn.init.normal_(raw, mean=0.0, std=std)
        shifted = raw - raw[-1, -1]
        self.weight_prefix.copy_(shifted.reshape(-1)[:-1])

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        weight = torch.cat(
            (self.weight_prefix, self.weight_prefix.new_zeros(1))
        ).view(self.num_embeddings, self.embedding_dim)
        if torch.is_grad_enabled():
            weight.retain_grad()
        self.full_weight = weight
        return F.embedding(idx, weight)


class TiedEmbeddingHead(nn.Module):
    """Parameter-free head applying the reconstructed tied embedding."""

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        # Match the discarded initialization draw made by nn.Linear before
        # its weight is tied, preserving all downstream initialization.
        discarded = torch.empty(out_features, in_features)
        nn.init.kaiming_uniform_(discarded, a=math.sqrt(5))

    def forward(
        self, x: torch.Tensor, weight: torch.Tensor
    ) -> torch.Tensor:
        return F.linear(x, weight)


class GaugeFixedPositionEmbedding(nn.Module):
>>>>>>> REPLACE

<<<<<<< SEARCH
class GaugeFixedTerminalLinear(nn.Module):
    """Linear layer with bias and three weight-column output gauges removed."""

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight_prefix = nn.ParameterList(
            [
                nn.Parameter(torch.empty(out_features - 1))
                for _ in range(3)
            ]
        )
        self.weight_rest = nn.Parameter(
            torch.empty(out_features, in_features - 3)
        )
=======
class GaugeFixedTerminalLinear(nn.Module):
    """Linear layer with bias and four weight-column output gauges removed."""

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight_prefix = nn.ParameterList(
            [
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
        self.ln1 = nn.LayerNorm(cfg.d_model)
=======
        self.ln1 = nn.LayerNorm(cfg.d_model, bias=False)
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.token_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
=======
        self.token_emb = GaugeFixedTiedEmbedding(
            cfg.vocab_size, cfg.d_model
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
        # Weight tying with input embeddings.
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.token_emb.weight

        self.apply(self._init_weights)
=======
        # The head consumes the same reconstructed learned weights as the
        # input embedding without registering a duplicate parameter.
        self.lm_head = TiedEmbeddingHead(
            cfg.d_model, cfg.vocab_size
        )

        self.apply(self._init_weights)
        # Match the final reset formerly performed through the tied Linear.
        self.token_emb.reset_parameters(std=0.02)
>>>>>>> REPLACE

<<<<<<< SEARCH
        if isinstance(module, GaugeFixedPositionEmbedding):
            module.reset_parameters(std=0.02)
=======
        if isinstance(module, GaugeFixedTiedEmbedding):
            module.reset_parameters(std=0.02)
        elif isinstance(module, GaugeFixedPositionEmbedding):
            module.reset_parameters(std=0.02)
>>>>>>> REPLACE

<<<<<<< SEARCH
                module.weight_rest.copy_(raw_weight[:, 3:])
                nn.init.zeros_(module.bias)
=======
                module.weight_rest.copy_(raw_weight[:, 4:])
                nn.init.zeros_(module.bias)
>>>>>>> REPLACE

<<<<<<< SEARCH
        logits = self.lm_head(x)
=======
        logits = self.lm_head(x, self.token_emb.full_weight)
>>>>>>> REPLACE

<<<<<<< SEARCH
    # Preserve full eight-coordinate AdamW dynamics for the positional,
    # terminal-bias, and three terminal-weight gauge vectors.
    gauge_params = [model.pos_emb.first]
=======
    # Preserve ambient AdamW dynamics for the tied-embedding, positional,
    # terminal-bias, and four terminal-weight gauge vectors.
    gauge_params = [
        model.token_emb.weight_prefix,
        model.pos_emb.first,
    ]
>>>>>>> REPLACE

<<<<<<< SEARCH
        full_gauge_grads = [model.pos_emb.full_first.grad.detach()]
=======
        full_gauge_grads = [
            model.token_emb.full_weight.grad.detach().reshape(-1),
            model.pos_emb.full_first.grad.detach(),
        ]
>>>>>>> REPLACE