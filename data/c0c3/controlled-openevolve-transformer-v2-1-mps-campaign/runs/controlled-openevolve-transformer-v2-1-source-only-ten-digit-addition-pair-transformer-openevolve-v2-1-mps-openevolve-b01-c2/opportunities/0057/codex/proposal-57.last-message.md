MECHANISM: Scalar token-position embedding shift gauge

HYPOTHESIS: Removing one redundant token-position shift coordinate will produce a 1,579-parameter model with at least 99% accuracy.

INTENDED_EDIT: Anchor position 0’s first coordinate, shift its initialized value into that coordinate of every tied token embedding, and retain all other embedding coefficients.

EVIDENCE: The current 1,580-parameter design achieved 99.83%; the eight-coordinate version collapsed to 70.62%, motivating the smallest one-coordinate ablation of that exact gauge.

<<<<<<< SEARCH
class TinyDecoderLM(nn.Module):
=======
class OneCoordinateGaugedPositionEmbedding(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim

        # Consume the same constructor RNG stream as nn.Embedding.
        base = nn.Embedding(num_embeddings, embedding_dim)
        retained = num_embeddings * embedding_dim - 1
        self.weight = nn.Parameter(base.weight.new_empty(retained))
        self.register_buffer(
            "_init_token_shift", base.weight.new_zeros(()), persistent=False
        )

    def forward(self, indices: torch.Tensor) -> torch.Tensor:
        weight = torch.cat((self.weight.new_zeros(1), self.weight)).view(
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
        self.pos_emb = OneCoordinateGaugedPositionEmbedding(
            cfg.max_seq_len, cfg.d_model
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.apply(self._init_weights)

    @staticmethod
=======
        self.apply(self._init_weights)

        # Preserve every initialized input embedding sum. The tied output
        # logits change only by a vocabulary-wide common offset.
        with torch.no_grad():
            self.token_emb.weight[:, 0].add_(self.pos_emb._init_token_shift)

    @staticmethod
>>>>>>> REPLACE

<<<<<<< SEARCH
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, LayerNormGaugedQKV):
=======
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, OneCoordinateGaugedPositionEmbedding):
            with torch.no_grad():
                full = module.weight.new_empty(
                    module.num_embeddings, module.embedding_dim
                )
                nn.init.normal_(full, mean=0.0, std=0.02)
                shift = full[0, 0].clone()
                full[:, 0].sub_(shift)
                module.weight.copy_(full.flatten()[1:])
                module._init_token_shift.copy_(shift)
        elif isinstance(module, LayerNormGaugedQKV):
>>>>>>> REPLACE