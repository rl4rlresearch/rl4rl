MECHANISM: Nonadjacent scalar token-position shift gauge

HYPOTHESIS: Extending the verified 1,579-parameter design by anchoring positional coordinate 3 will produce a 1,578-parameter model with at least 99% accuracy.

INTENDED_EDIT: Reproduce the seven verified QKV row gauges and anchor positional coordinates 0 and 3, transferring both initialized shifts into the tied token embeddings.

EVIDENCE: Coordinate 0 achieved 99.79% at 1,579 parameters, whereas adjacent coordinate 1 collapsed. Coordinate 3 is the first coordinate with an anchored `ln1` scale, making it the most informative nonadjacent test of whether the failure was coordinate-specific.

<<<<<<< SEARCH
class LayerNormGaugedQKV(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.in_features = d_model
        self.out_features = 3 * d_model
        self.gauged_rows = (0, 1)
=======
class LayerNormGaugedQKV(nn.Module):
    def __init__(self, d_model: int, head_dim: int):
        super().__init__()
        self.in_features = d_model
        self.out_features = 3 * d_model
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
        # Retain query bias while gauge-fixing two normalized-input QKV rows.
        self.qkv = LayerNormGaugedQKV(d_model)
=======
        # Retain query bias while applying the verified query, key, and value
        # normalized-input gauges.
        self.qkv = LayerNormGaugedQKV(d_model, self.head_dim)
>>>>>>> REPLACE

<<<<<<< SEARCH
class TinyDecoderLM(nn.Module):
    def __init__(self, cfg: ModelConfig):
=======
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
        # Flat indices 0 and 3 are position zero's coordinates 0 and 3.
        flat = torch.cat(
            (
                self.weight.new_zeros(1),
                self.weight[:2],
                self.weight.new_zeros(1),
                self.weight[2:],
            )
        )
        return F.embedding(
            indices, flat.view(self.num_embeddings, self.embedding_dim)
        )


class TinyDecoderLM(nn.Module):
    def __init__(self, cfg: ModelConfig):
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.token_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_emb = nn.Embedding(cfg.max_seq_len, cfg.d_model)
        self.drop = nn.Dropout(cfg.dropout)
=======
        self.token_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_emb = TwoCoordinateGaugedPositionEmbedding(
            cfg.max_seq_len, cfg.d_model
        )
        self.drop = nn.Dropout(cfg.dropout)
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, LayerNormGaugedQKV):
=======
        self.apply(self._init_weights)

        # Preserve every initialized input embedding sum. Because the output
        # embedding is tied, each shift changes logits only by a common offset.
        with torch.no_grad():
            self.token_emb.weight[:, 0].add_(
                self.pos_emb._init_token_shift[0]
            )
            self.token_emb.weight[:, 3].add_(
                self.pos_emb._init_token_shift[1]
            )

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, TwoCoordinateGaugedPositionEmbedding):
            with torch.no_grad():
                full = module.weight.new_empty(
                    module.num_embeddings, module.embedding_dim
                )
                nn.init.normal_(full, mean=0.0, std=0.02)
                shifts = torch.stack((full[0, 0], full[0, 3])).clone()
                full[:, 0].sub_(shifts[0])
                full[:, 3].sub_(shifts[1])
                flat = full.flatten()
                module.weight.copy_(
                    torch.cat((flat[1:3], flat[4:]))
                )
                module._init_token_shift.copy_(shifts)
        elif isinstance(module, LayerNormGaugedQKV):
>>>>>>> REPLACE