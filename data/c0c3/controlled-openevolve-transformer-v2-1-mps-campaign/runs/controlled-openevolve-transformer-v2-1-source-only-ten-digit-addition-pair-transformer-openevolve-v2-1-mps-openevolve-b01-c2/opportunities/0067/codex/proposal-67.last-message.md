MECHANISM: Mean-centered positional translation gauge

HYPOTHESIS: Reparameterizing positional coordinate 4 with zero mean, on top of the verified 1,574-parameter design, will yield 1,573 parameters while maintaining at least 99% accuracy.

INTENDED_EDIT: Reproduce the verified balanced query-bias, coordinate-7 positional, and tied-token translation gauges, then remove one additional positional coefficient by centering coordinate 4 across positions and transferring its mean into every token embedding.

EVIDENCE: The 1,574-parameter reference reached 99.88%; although anchoring position zero’s coordinate 4 collapsed, the exact translation gauge remains available, and a symmetric zero-mean chart avoids privileging the position-zero coefficient implicated by that failure.

<<<<<<< SEARCH
        # Consume the same constructor RNG stream as the replaced nn.Linear.
        base = nn.Linear(d_model, 3 * d_model)
        retained = self.out_features * d_model - len(self.gauged_rows)
        self.weight = nn.Parameter(base.weight.new_empty(retained))
        self.bias = nn.Parameter(base.bias.new_empty(d_model))
=======
        # Consume the same constructor RNG stream as the replaced nn.Linear.
        base = nn.Linear(d_model, 3 * d_model)
        retained = self.out_features * d_model - len(self.gauged_rows)
        self.weight = nn.Parameter(base.weight.new_empty(retained))
        # Fix local query coordinate 2 in both heads.
        self.fixed_query_biases = (2, head_dim + 2)
        self.bias = nn.Parameter(
            base.bias.new_empty(d_model - len(self.fixed_query_biases))
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
        fused_bias = torch.cat(
            (
                self.bias,
                self.bias.new_zeros(self.in_features),
                self.bias.new_zeros(self.in_features),
            )
        )
        return F.linear(x, weight, fused_bias)
=======
        query_bias_parts = []
        bias_index = 0
        for coordinate in range(self.in_features):
            if coordinate in self.fixed_query_biases:
                query_bias_parts.append(self.bias.new_zeros(()))
            else:
                query_bias_parts.append(self.bias[bias_index])
                bias_index += 1
        query_bias = torch.stack(query_bias_parts)

        fused_bias = torch.cat(
            (
                query_bias,
                self.bias.new_zeros(self.in_features),
                self.bias.new_zeros(self.in_features),
            )
        )
        return F.linear(x, weight, fused_bias)
>>>>>>> REPLACE

<<<<<<< SEARCH
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
=======
class ScalarGaugedTokenEmbedding(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim

        # Consume the same constructor RNG stream as nn.Embedding.
        base = nn.Embedding(num_embeddings, embedding_dim)
        self.weight = nn.Parameter(
            base.weight.new_empty(num_embeddings * embedding_dim - 1)
        )

    def dense_weight(self) -> torch.Tensor:
        flat = torch.cat((self.weight.new_zeros(1), self.weight))
        return flat.view(self.num_embeddings, self.embedding_dim)

    def forward(self, indices: torch.Tensor) -> torch.Tensor:
        return F.embedding(indices, self.dense_weight())

    @torch.no_grad()
    def transfer_coordinate_shifts(self, coordinates, shifts) -> None:
        full = self.dense_weight().clone()
        for coordinate, shift in zip(coordinates, shifts):
            full[:, coordinate].add_(shift)

        # A common scalar translation is invisible to the residual stream's
        # LayerNorms and contributes only a vocabulary-common logit offset.
        anchor = full[0, 0].clone()
        full.sub_(anchor)
        self.weight.copy_(full.flatten()[1:])


class TiedGaugedLMHead(nn.Module):
    def __init__(self, embedding: ScalarGaugedTokenEmbedding):
        super().__init__()
        self.in_features = embedding.embedding_dim
        self.out_features = embedding.num_embeddings

        # Consume the constructor RNG stream of the replaced tied nn.Linear.
        base = nn.Linear(self.in_features, self.out_features, bias=False)
        object.__setattr__(self, "_embedding", embedding)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self._embedding.dense_weight())


class CenteredCoordinateGaugedPositionEmbedding(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.anchor_coordinates = (0, 3, 7)
        self.centered_coordinate = 4

        # Consume the same constructor RNG stream as nn.Embedding.
        base = nn.Embedding(num_embeddings, embedding_dim)
        retained = num_embeddings * embedding_dim - 4
        self.weight = nn.Parameter(base.weight.new_empty(retained))
        self.other_count = (
            num_embeddings * (embedding_dim - 1)
            - len(self.anchor_coordinates)
        )
        self.register_buffer(
            "_init_token_shift", base.weight.new_zeros(4), persistent=False
        )

    def dense_weight(self) -> torch.Tensor:
        other = self.weight[:self.other_count]
        centered_free = self.weight[self.other_count:]
        centered = torch.cat(
            (centered_free, (-centered_free.sum()).unsqueeze(0))
        )

        values = []
        other_index = 0
        for position in range(self.num_embeddings):
            for coordinate in range(self.embedding_dim):
                if coordinate == self.centered_coordinate:
                    values.append(centered[position])
                elif position == 0 and coordinate in self.anchor_coordinates:
                    values.append(self.weight.new_zeros(()))
                else:
                    values.append(other[other_index])
                    other_index += 1
        return torch.stack(values).view(
            self.num_embeddings, self.embedding_dim
        )

    def forward(self, indices: torch.Tensor) -> torch.Tensor:
        return F.embedding(indices, self.dense_weight())
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.cfg = cfg
        self.token_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_emb = TwoCoordinateGaugedPositionEmbedding(
            cfg.max_seq_len, cfg.d_model
        )
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.ln_f = nn.LayerNorm(cfg.d_model)

        # Weight tying with input embeddings.
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.token_emb.weight

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
=======
        self.cfg = cfg
        self.token_emb = ScalarGaugedTokenEmbedding(
            cfg.vocab_size, cfg.d_model
        )
        self.pos_emb = CenteredCoordinateGaugedPositionEmbedding(
            cfg.max_seq_len, cfg.d_model
        )
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.ln_f = nn.LayerNorm(cfg.d_model)

        # Reconstruct input and output weights from one shared parameter.
        self.lm_head = TiedGaugedLMHead(self.token_emb)

        self.apply(self._init_weights)

        # Restore token-plus-position inputs after fixing three position-zero
        # coordinates and centering the fourth positional coordinate.
        self.token_emb.transfer_coordinate_shifts(
            (0, 3, 4, 7), self.pos_emb._init_token_shift
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
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
=======
        if isinstance(module, ScalarGaugedTokenEmbedding):
            with torch.no_grad():
                full = module.weight.new_empty(
                    module.num_embeddings, module.embedding_dim
                )
                nn.init.normal_(full, mean=0.0, std=0.02)
                anchor = full[0, 0].clone()
                full.sub_(anchor)
                module.weight.copy_(full.flatten()[1:])
        elif isinstance(module, TiedGaugedLMHead):
            with torch.no_grad():
                embedding = module._embedding
                full = embedding.weight.new_empty(
                    embedding.num_embeddings, embedding.embedding_dim
                )
                nn.init.normal_(full, mean=0.0, std=0.02)
                anchor = full[0, 0].clone()
                full.sub_(anchor)
                embedding.weight.copy_(full.flatten()[1:])
        elif isinstance(module, CenteredCoordinateGaugedPositionEmbedding):
            with torch.no_grad():
                full = module.weight.new_empty(
                    module.num_embeddings, module.embedding_dim
                )
                nn.init.normal_(full, mean=0.0, std=0.02)

                shift0 = full[0, 0].clone()
                shift3 = full[0, 3].clone()
                shift7 = full[0, 7].clone()
                shift4 = full[:, 4].mean().clone()
                full[:, 0].sub_(shift0)
                full[:, 3].sub_(shift3)
                full[:, 4].sub_(shift4)
                full[:, 7].sub_(shift7)

                other = []
                for position in range(module.num_embeddings):
                    for coordinate in range(module.embedding_dim):
                        if coordinate == module.centered_coordinate:
                            continue
                        if (
                            position == 0
                            and coordinate in module.anchor_coordinates
                        ):
                            continue
                        other.append(full[position, coordinate])
                module.weight.copy_(
                    torch.cat(
                        (
                            torch.stack(other),
                            full[:-1, module.centered_coordinate],
                        )
                    )
                )
                module._init_token_shift.copy_(
                    torch.stack((shift0, shift3, shift4, shift7))
                )
        elif isinstance(module, LayerNormGaugedQKV):
>>>>>>> REPLACE