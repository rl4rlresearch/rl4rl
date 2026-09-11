MECHANISM: Fifth token–position translation quotient

HYPOTHESIS: Extending the verified 1,535-parameter design by anchoring position-zero coordinate 2 and transferring its shift into the tied token embedding will produce a 1,534-parameter model with at least 99% accuracy.

INTENDED_EDIT: Reproduce all verified query, key, and value-row gauges, retain dense-coordinate AdamW for sensitive rows 15, 20, and 23, and extend the positional quotient from coordinates 0, 1, 3, and 7 to coordinates 0, 1, 2, 3, and 7.

EVIDENCE: The fourth positional anchor produced 99.82% accuracy at 1,535 parameters, while the underlying query-row-7 design achieved 99.96% at 1,536 parameters; coordinate 2 applies the same exact token–position translation gauge for one additional reduction.

<<<<<<< SEARCH
        # Retain the verified balanced query gauges, gauge every key row and
        # every first-head value row, then gauge the first two value rows of
        # the second head. Row 21 uses the ordinary reduced-coordinate update.
        self.gauged_rows = (
            0,
            1,
            head_dim,
            head_dim + 1,
            d_model,
            d_model + 1,
            d_model + 2,
            d_model + 3,
            d_model + head_dim,
            d_model + head_dim + 1,
            d_model + head_dim + 2,
            d_model + head_dim + 3,
            2 * d_model,
            2 * d_model + 1,
            2 * d_model + 2,
            2 * d_model + 3,
            2 * d_model + head_dim,
            2 * d_model + head_dim + 1,
        )
=======
        # Retain the verified query-row-7 design and every key and value
        # gauge. Rows 15, 20, and 23 use recovered dense-coordinate AdamW.
        self.gauged_rows = (
            0,
            1,
            2,
            3,
            head_dim,
            head_dim + 1,
            head_dim + 3,
            d_model,
            d_model + 1,
            d_model + 2,
            d_model + 3,
            d_model + head_dim,
            d_model + head_dim + 1,
            d_model + head_dim + 2,
            d_model + head_dim + 3,
            2 * d_model,
            2 * d_model + 1,
            2 * d_model + 2,
            2 * d_model + 3,
            2 * d_model + head_dim,
            2 * d_model + head_dim + 1,
            2 * d_model + head_dim + 2,
            2 * d_model + head_dim + 3,
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
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
=======
class FiveCoordinateGaugedPositionEmbedding(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim

        # Consume the same constructor RNG stream as nn.Embedding.
        base = nn.Embedding(num_embeddings, embedding_dim)
        retained = num_embeddings * embedding_dim - 5
        self.weight = nn.Parameter(base.weight.new_empty(retained))
        self.register_buffer(
            "_init_token_shift", base.weight.new_zeros(5), persistent=False
        )

    def forward(self, indices: torch.Tensor) -> torch.Tensor:
        # Anchor position zero's coordinates 0, 1, 2, 3, and 7.
        flat = torch.cat(
            (
                self.weight.new_zeros(4),
                self.weight[:3],
                self.weight.new_zeros(1),
                self.weight[3:],
            )
        )
        return F.embedding(
            indices, flat.view(self.num_embeddings, self.embedding_dim)
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.pos_emb = ThreeCoordinateGaugedPositionEmbedding(
            cfg.max_seq_len, cfg.d_model
        )
=======
        self.pos_emb = FiveCoordinateGaugedPositionEmbedding(
            cfg.max_seq_len, cfg.d_model
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.token_emb.transfer_coordinate_shifts(
            (0, 3, 7), self.pos_emb._init_token_shift
        )
=======
        self.token_emb.transfer_coordinate_shifts(
            (0, 1, 2, 3, 7), self.pos_emb._init_token_shift
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
        elif isinstance(module, ThreeCoordinateGaugedPositionEmbedding):
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
=======
        elif isinstance(module, FiveCoordinateGaugedPositionEmbedding):
            with torch.no_grad():
                full = module.weight.new_empty(
                    module.num_embeddings, module.embedding_dim
                )
                nn.init.normal_(full, mean=0.0, std=0.02)
                shifts = torch.stack(
                    (
                        full[0, 0],
                        full[0, 1],
                        full[0, 2],
                        full[0, 3],
                        full[0, 7],
                    )
                ).clone()
                full[:, 0].sub_(shifts[0])
                full[:, 1].sub_(shifts[1])
                full[:, 2].sub_(shifts[2])
                full[:, 3].sub_(shifts[3])
                full[:, 7].sub_(shifts[4])
                flat = full.flatten()
                module.weight.copy_(
                    torch.cat((flat[4:7], flat[8:]))
                )
                module._init_token_shift.copy_(shifts)
>>>>>>> REPLACE

<<<<<<< SEARCH
    # Residual-output quotients retain dense output-coordinate moments. The
    # completed final key row and new second-head value row additionally use
    # dense moments for their omitted normalized-input coefficients.
=======
    # Residual-output quotients retain dense output-coordinate moments. Key
    # row 15 and value rows 20 and 23 retain their verified dense-coordinate
    # updates; the remaining gauged QKV rows use reduced-coordinate AdamW.
>>>>>>> REPLACE

<<<<<<< SEARCH
                (
                    2 * qkv.in_features - 1,
                    2 * qkv.in_features + block.attn.head_dim,
                ),
=======
                (
                    2 * qkv.in_features - 1,
                    2 * qkv.in_features + block.attn.head_dim,
                    3 * qkv.in_features - 1,
                ),
>>>>>>> REPLACE