MECHANISM: Token-position embedding shift gauge

HYPOTHESIS: Combining the verified four-row balanced QKV gauge with an exact 8-parameter token-position embedding shift gauge will produce a 1,575-parameter model with at least 99% accuracy.

INTENDED_EDIT: Reproduce query-row gauges 0, 1, 4, and 5, then anchor positional row 0 to zero while shifting its initialized vector into every tied token embedding.

EVIDENCE: Reference Design 3 reached 99.98% accuracy at 1,583 parameters. Anchoring one positional row removes eight redundant parameters without reducing the represented distributions: shifting that row into all token embeddings preserves every input embedding sum, while the tied output logits change only by a vocabulary-wide common offset.

<<<<<<< SEARCH
    vocab_size: int


class LayerNormGaugedQKV(nn.Module):
=======
    vocab_size: int


class AnchoredPositionEmbedding(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim

        # Consume the same constructor RNG stream as the replaced embedding.
        base = nn.Embedding(num_embeddings, embedding_dim)
        self.weight = nn.Parameter(
            base.weight.new_empty(num_embeddings - 1, embedding_dim)
        )

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        weight = torch.cat(
            (self.weight.new_zeros(1, self.embedding_dim), self.weight), dim=0
        )
        return F.embedding(idx, weight)


class LayerNormGaugedQKV(nn.Module):
>>>>>>> REPLACE

<<<<<<< SEARCH
class LayerNormGaugedQKV(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.in_features = d_model
        self.out_features = 3 * d_model
        self.gauged_rows = (0,)
        self.ungauged_rows = tuple(range(1, self.out_features))
=======
class LayerNormGaugedQKV(nn.Module):
    def __init__(self, d_model: int, head_dim: int):
        super().__init__()
        self.in_features = d_model
        self.out_features = 3 * d_model
        self.gauged_rows = (0, 1, head_dim, head_dim + 1)
        self.ungauged_rows = tuple(
            row for row in range(self.out_features) if row not in self.gauged_rows
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
        row_width = self.in_features - 1
        gauged = torch.cat(
            (
                self.weight[:row_width].view(1, row_width),
                self.weight.new_zeros(1, 1),
            ),
            dim=1,
        )
        ungauged = self.weight[row_width:].view(
            len(self.ungauged_rows), self.in_features
        )
        weight = torch.cat((gauged, ungauged), dim=0)
=======
        row_width = self.in_features - 1
        split = len(self.gauged_rows) * row_width
        gauged = torch.cat(
            (
                self.weight[:split].view(len(self.gauged_rows), row_width),
                self.weight.new_zeros(len(self.gauged_rows), 1),
            ),
            dim=1,
        )
        ungauged = self.weight[split:].view(
            len(self.ungauged_rows), self.in_features
        )

        rows = []
        gauged_index = 0
        ungauged_index = 0
        for row in range(self.out_features):
            if row in self.gauged_rows:
                rows.append(gauged[gauged_index])
                gauged_index += 1
            else:
                rows.append(ungauged[ungauged_index])
                ungauged_index += 1
        weight = torch.stack(rows)
>>>>>>> REPLACE

<<<<<<< SEARCH
        # Retain query bias and one representative of QKV row 0's normalized-
        # input coefficient gauge.
        self.qkv = LayerNormGaugedQKV(d_model)
=======
        # Retain query bias while gauge-fixing two normalized-input query rows
        # in each attention head.
        self.qkv = LayerNormGaugedQKV(d_model, self.head_dim)
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.token_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_emb = nn.Embedding(cfg.max_seq_len, cfg.d_model)
=======
        self.token_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_emb = AnchoredPositionEmbedding(cfg.max_seq_len, cfg.d_model)
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.apply(self._init_weights)

    @staticmethod
=======
        self.apply(self._init_weights)

        # Move the omitted positional anchor into every tied token vector.
        # Inputs are unchanged, and output logits gain only a common offset.
        with torch.no_grad():
            self.token_emb.weight.add_(self.pos_emb._initial_anchor)
        del self.pos_emb._initial_anchor

    @staticmethod
>>>>>>> REPLACE

<<<<<<< SEARCH
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, LayerNormGaugedQKV):
=======
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, AnchoredPositionEmbedding):
            with torch.no_grad():
                full = module.weight.new_empty(
                    module.num_embeddings, module.embedding_dim
                )
                nn.init.normal_(full, mean=0.0, std=0.02)
                module.weight.copy_(full[1:] - full[:1])
                module._initial_anchor = full[0].clone()
        elif isinstance(module, LayerNormGaugedQKV):
>>>>>>> REPLACE