MECHANISM: Third scalar token-position embedding shift gauge

HYPOTHESIS: Extending the verified 1,578-parameter design by anchoring positional coordinate 4 will produce a 1,577-parameter model with at least 99% accuracy.

INTENDED_EDIT: Reproduce the seven verified QKV row gauges and anchor positional coordinates 0, 3, and 4, transferring their initialized shifts into the tied token embeddings.

EVIDENCE: Reference Design 2 achieved 99.91% accuracy at 1,578 parameters with coordinates 0 and 3 anchored; coordinate 4 is also backed by a fixed `ln1` scale, making it the closest informative extension while avoiding the failed unanchored coordinate-1 choice.

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
        self.ungauged_rows = tuple(
            row for row in range(self.out_features) if row not in self.gauged_rows
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
    def forward(self, x: torch.Tensor) -> torch.Tensor:
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

        fused_bias = torch.cat(
=======
    def forward(self, x: torch.Tensor) -> torch.Tensor:
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

        fused_bias = torch.cat(
>>>>>>> REPLACE

<<<<<<< SEARCH
        # Retain query bias and one representative of QKV row 0's normalized-
        # input coefficient gauge.
        self.qkv = LayerNormGaugedQKV(d_model)
=======
        # Retain query bias while applying the verified query, key, and value
        # normalized-input gauges.
        self.qkv = LayerNormGaugedQKV(d_model, self.head_dim)
>>>>>>> REPLACE

<<<<<<< SEARCH
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class TinyDecoderLM(nn.Module):
=======
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class ThreeCoordinateGaugedPositionEmbedding(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim

        # Consume the same constructor RNG stream as nn.Embedding.
        base = nn.Embedding(num_embeddings, embedding_dim)
        retained = num_embeddings * embedding_dim - 3
        self.weight = nn.Parameter(base.weight.new_empty(retained))
        self.register_buffer(
            "_init_token_shift", base.weight.new_zeros(3), persistent=False
        )

    def forward(self, indices: torch.Tensor) -> torch.Tensor:
        # Flat indices 0, 3, and 4 are position zero's selected coordinates.
        flat = torch.cat(
            (
                self.weight.new_zeros(1),
                self.weight[:2],
                self.weight.new_zeros(2),
                self.weight[2:],
            )
        )
        return F.embedding(
            indices, flat.view(self.num_embeddings, self.embedding_dim)
        )


class TinyDecoderLM(nn.Module):
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.token_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_emb = nn.Embedding(cfg.max_seq_len, cfg.d_model)
        self.drop = nn.Dropout(cfg.dropout)
=======
        self.token_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_emb = ThreeCoordinateGaugedPositionEmbedding(
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
            self.token_emb.weight[:, 4].add_(
                self.pos_emb._init_token_shift[2]
            )

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, ThreeCoordinateGaugedPositionEmbedding):
            with torch.no_grad():
                full = module.weight.new_empty(
                    module.num_embeddings, module.embedding_dim
                )
                nn.init.normal_(full, mean=0.0, std=0.02)
                shifts = torch.stack(
                    (full[0, 0], full[0, 3], full[0, 4])
                ).clone()
                full[:, 0].sub_(shifts[0])
                full[:, 3].sub_(shifts[1])
                full[:, 4].sub_(shifts[2])
                flat = full.flatten()
                module.weight.copy_(
                    torch.cat((flat[1:3], flat[5:]))
                )
                module._init_token_shift.copy_(shifts)
        elif isinstance(module, LayerNormGaugedQKV):
>>>>>>> REPLACE