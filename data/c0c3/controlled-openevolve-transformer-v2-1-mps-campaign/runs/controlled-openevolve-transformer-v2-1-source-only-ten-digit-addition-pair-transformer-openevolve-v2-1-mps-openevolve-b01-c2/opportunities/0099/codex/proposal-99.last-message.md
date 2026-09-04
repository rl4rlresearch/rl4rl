MECHANISM: Additional first-head query-bias basis quotient

HYPOTHESIS: Extending the verified 1,535-parameter design by fixing first-head query-bias coordinate 3 will produce a 1,534-parameter model with at least 99% accuracy.

INTENDED_EDIT: Reproduce the verified query-row-7 and four-position-anchor design, retain dense-coordinate AdamW for QKV rows 15, 20, and 23, and remove one additional first-head query-bias parameter.

EVIDENCE: The 1,535-parameter design achieved 99.82% while already fixing one query-bias coordinate per head; testing another coordinate of that successful attention-basis quotient is more informative than positional coordinates 2 and 4 or a sixth LayerNorm scale, which all failed.

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
        # Retain every verified query, key, and value-row gauge. Sensitive
        # rows 15, 20, and 23 use recovered dense-coordinate AdamW moments.
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
        self.fixed_query_biases = (2, head_dim + 2)
        self.bias = nn.Parameter(
            base.bias.new_empty(d_model - len(self.fixed_query_biases))
        )
=======
        # Use the query-key basis freedom to fix a second coordinate in the
        # robust first head while retaining the verified second-head anchor.
        self.fixed_query_biases = (2, 3, head_dim + 2)
        self.bias = nn.Parameter(
            base.bias.new_empty(d_model - len(self.fixed_query_biases))
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
=======
class FourCoordinateGaugedPositionEmbedding(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim

        # Consume the same constructor RNG stream as nn.Embedding.
        base = nn.Embedding(num_embeddings, embedding_dim)
        retained = num_embeddings * embedding_dim - 4
        self.weight = nn.Parameter(base.weight.new_empty(retained))
        self.register_buffer(
            "_init_token_shift", base.weight.new_zeros(4), persistent=False
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
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
=======
        flat = torch.cat(
            (
                self.weight.new_zeros(2),
                self.weight[:1],
                self.weight.new_zeros(1),
                self.weight[1:4],
                self.weight.new_zeros(1),
                self.weight[4:],
            )
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.pos_emb = ThreeCoordinateGaugedPositionEmbedding(
            cfg.max_seq_len, cfg.d_model
        )
=======
        self.pos_emb = FourCoordinateGaugedPositionEmbedding(
            cfg.max_seq_len, cfg.d_model
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.token_emb.transfer_coordinate_shifts(
            (0, 3, 7), self.pos_emb._init_token_shift
        )
=======
        self.token_emb.transfer_coordinate_shifts(
            (0, 1, 3, 7), self.pos_emb._init_token_shift
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
        elif isinstance(module, FourCoordinateGaugedPositionEmbedding):
            with torch.no_grad():
                full = module.weight.new_empty(
                    module.num_embeddings, module.embedding_dim
                )
                nn.init.normal_(full, mean=0.0, std=0.02)
                shifts = torch.stack(
                    (full[0, 0], full[0, 1], full[0, 3], full[0, 7])
                ).clone()
                full[:, 0].sub_(shifts[0])
                full[:, 1].sub_(shifts[1])
                full[:, 3].sub_(shifts[2])
                full[:, 7].sub_(shifts[3])
                flat = full.flatten()
                module.weight.copy_(
                    torch.cat((flat[2:3], flat[4:7], flat[8:]))
                )
                module._init_token_shift.copy_(shifts)
>>>>>>> REPLACE

<<<<<<< SEARCH
    # Residual-output quotients retain dense output-coordinate moments. The
    # completed final key row and new second-head value row additionally use
    # dense moments for their omitted normalized-input coefficients.
=======
    # Residual-output quotients retain dense output-coordinate moments. The
    # optimization-sensitive key row 15 and value rows 20 and 23 additionally
    # use dense moments for their omitted normalized-input coefficients.
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