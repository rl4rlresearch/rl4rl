MECHANISM: Global tied-token embedding translation gauge

HYPOTHESIS: Adding an exact scalar token-embedding translation gauge to the verified 1,575-parameter balanced query-bias design will produce a 1,574-parameter model with at least 99% accuracy.

INTENDED_EDIT: Reproduce the successful query-bias constraints at coordinates 2 and 6, then anchor the tied token embedding’s first scalar and reconstruct both input embeddings and output logits from the remaining coefficients while preserving initialization modulo LayerNorm and softmax invariances.

EVIDENCE: Fixing query-bias coordinate 2 in both heads achieved 99.83% at 1,575 parameters, whereas constraining query-bias coordinate 3 already fell to 98.48%; this motivates an orthogonal exact translation gauge instead of another query-bias restriction.

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
        # Fix local query coordinate 2 in both heads. Key and value biases
        # remain absent through their existing exact attention gauges.
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
class ThreeCoordinateGaugedPositionEmbedding(nn.Module):
=======
class ScalarGaugedTokenEmbedding(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim

        # Consume the same constructor RNG stream as nn.Embedding.
        base = nn.Embedding(num_embeddings, embedding_dim)
        retained = num_embeddings * embedding_dim - 1
        self.weight = nn.Parameter(base.weight.new_empty(retained))

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

        # A common scalar translation of every embedding coordinate is
        # invisible to all LayerNorms and adds only a common output-logit shift.
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


class ThreeCoordinateGaugedPositionEmbedding(nn.Module):
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.token_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
=======
        self.token_emb = ScalarGaugedTokenEmbedding(
            cfg.vocab_size, cfg.d_model
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
        # Weight tying with input embeddings.
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.token_emb.weight
=======
        # Reconstruct input and output weights from one shared gauged parameter.
        self.lm_head = TiedGaugedLMHead(self.token_emb)
>>>>>>> REPLACE

<<<<<<< SEARCH
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
=======
        # Transfer the positional anchors, then restore the token-embedding
        # translation gauge without changing normalized states or predictions.
        self.token_emb.transfer_coordinate_shifts(
            (0, 3, 7), self.pos_emb._init_token_shift
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, ThreeCoordinateGaugedPositionEmbedding):
=======
    @staticmethod
    def _init_weights(module: nn.Module) -> None:
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
        elif isinstance(module, ThreeCoordinateGaugedPositionEmbedding):
>>>>>>> REPLACE