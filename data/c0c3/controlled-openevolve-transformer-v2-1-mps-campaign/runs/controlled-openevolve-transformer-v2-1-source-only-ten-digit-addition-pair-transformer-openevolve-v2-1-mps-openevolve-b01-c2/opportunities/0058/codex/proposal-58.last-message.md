MECHANISM: Second scalar token-position embedding shift gauge

HYPOTHESIS: Extending the verified one-coordinate positional gauge to a second coordinate will produce a 1,578-parameter model with at least 99% accuracy.

INTENDED_EDIT: Reproduce the verified query, key, and two value-row gauges, then anchor position 0’s first two coordinates and transfer their initialized shifts into the tied token embeddings.

EVIDENCE: The one-coordinate positional gauge achieved 99.79% accuracy at 1,579 parameters, while gauging all eight coordinates collapsed; adding only the adjacent second coordinate is the smallest informative extension, and further QKV-row gauges have already failed.

<<<<<<< SEARCH
        # Balance four query-row gauges across the two attention heads, then
        # gauge the first key row without further constraining either query.
        self.gauged_rows = (0, 1, head_dim, head_dim + 1, d_model)
=======
        # Retain the verified balanced query and first-key gauges, then gauge
        # two adjacent value coordinates in the first head.
        self.gauged_rows = (
            0,
            1,
            head_dim,
            head_dim + 1,
            d_model,
            2 * d_model,
            2 * d_model + 1,
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
        x = x + self.mlp(self.ln2(x))
        return x


class TinyDecoderLM(nn.Module):
=======
        x = x + self.mlp(self.ln2(x))
        return x


class TwoCoordinateGaugedPositionEmbedding(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim

        # Consume the same constructor RNG stream as nn.Embedding.
        base = nn.Embedding(num_embeddings, embedding_dim)
        retained = num_embeddings * embedding_dim - 2
        self.weight = nn.Parameter(base.weight.new_empty(retained))
        self.register_buffer(
            "_init_token_shift", base.weight.new_zeros(2), persistent=False
        )

    def forward(self, indices: torch.Tensor) -> torch.Tensor:
        weight = torch.cat((self.weight.new_zeros(2), self.weight)).view(
            self.num_embeddings, self.embedding_dim
        )
        return F.embedding(indices, weight)


class TinyDecoderLM(nn.Module):
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.token_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_emb = nn.Embedding(cfg.max_seq_len, cfg.d_model)
=======
        self.token_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_emb = TwoCoordinateGaugedPositionEmbedding(
            cfg.max_seq_len, cfg.d_model
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.apply(self._init_weights)

    @staticmethod
=======
        self.apply(self._init_weights)

        # Preserve initialized input sums; tied output logits change only by
        # a vocabulary-wide common offset.
        with torch.no_grad():
            self.token_emb.weight[:, :2].add_(self.pos_emb._init_token_shift)

    @staticmethod
>>>>>>> REPLACE

<<<<<<< SEARCH
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, LayerNormGaugedQKV):
=======
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, TwoCoordinateGaugedPositionEmbedding):
            with torch.no_grad():
                full = module.weight.new_empty(
                    module.num_embeddings, module.embedding_dim
                )
                nn.init.normal_(full, mean=0.0, std=0.02)
                shift = full[0, :2].clone()
                full[:, :2].sub_(shift)
                module.weight.copy_(full.flatten()[2:])
                module._init_token_shift.copy_(shift)
        elif isinstance(module, LayerNormGaugedQKV):
>>>>>>> REPLACE