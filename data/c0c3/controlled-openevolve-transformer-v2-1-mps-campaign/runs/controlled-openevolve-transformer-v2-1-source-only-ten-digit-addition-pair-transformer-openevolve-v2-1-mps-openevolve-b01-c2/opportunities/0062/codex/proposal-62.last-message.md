MECHANISM: Third scalar token-position shift gauge on the normalized-input pivot coordinate

HYPOTHESIS: Extending the verified 1,578-parameter coordinates-0-and-3 design by anchoring positional coordinate 7 will produce a 1,577-parameter model with at least 99% accuracy.

INTENDED_EDIT: Reproduce the successful positional anchors at coordinates 0 and 3, then also anchor coordinate 7 and transfer all three initialized shifts into the tied token embeddings.

EVIDENCE: Coordinates 0 and 3 achieved 99.91% accuracy at 1,578 parameters, while coordinate 4 fell to 72.81%. Coordinate 7 is the strongest untested alternative because it is the fixed-scale input coordinate omitted by every existing normalized-input QKV and MLP row gauge.

<<<<<<< SEARCH
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
=======
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
        # Flat indices 0, 3, and 7 are position zero's selected coordinates.
        flat = torch.cat(
            (
                self.weight.new_zeros(1),
                self.weight[:2],
                self.weight.new_zeros(1),
                self.weight[2:5],
                self.weight.new_zeros(1),
                self.weight[5:],
            )
        )
        return F.embedding(
            indices, flat.view(self.num_embeddings, self.embedding_dim)
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.pos_emb = OneCoordinateGaugedPositionEmbedding(
            cfg.max_seq_len, cfg.d_model
        )
=======
        self.pos_emb = ThreeCoordinateGaugedPositionEmbedding(
            cfg.max_seq_len, cfg.d_model
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
        # Preserve every initialized input embedding sum. The tied output
        # logits change only by a vocabulary-wide common offset.
        with torch.no_grad():
            self.token_emb.weight[:, 0].add_(self.pos_emb._init_token_shift)
=======
        # Preserve every initialized input embedding sum. With tied output
        # embeddings, each shift changes logits only by a common offset.
        with torch.no_grad():
            self.token_emb.weight[:, 0].add_(
                self.pos_emb._init_token_shift[0]
            )
            self.token_emb.weight[:, 3].add_(
                self.pos_emb._init_token_shift[1]
            )
            self.token_emb.weight[:, 7].add_(
                self.pos_emb._init_token_shift[2]
            )
>>>>>>> REPLACE

<<<<<<< SEARCH
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
=======
        if isinstance(module, ThreeCoordinateGaugedPositionEmbedding):
            with torch.no_grad():
                full = module.weight.new_empty(
                    module.num_embeddings, module.embedding_dim
                )
                nn.init.normal_(full, mean=0.0, std=0.02)
                shifts = torch.stack(
                    (full[0, 0], full[0, 3], full[0, 7])
                ).clone()
                full[:, 0].sub_(shifts[0])
                full[:, 3].sub_(shifts[1])
                full[:, 7].sub_(shifts[2])
                flat = full.flatten()
                module.weight.copy_(
                    torch.cat((flat[1:3], flat[4:7], flat[8:]))
                )
                module._init_token_shift.copy_(shifts)
>>>>>>> REPLACE