MECHANISM: Tied-softmax token/position common-mode quotient

HYPOTHESIS: Constraining the fifth token-content coordinate to be mean-free across vocabulary will reduce the model from 1,554 to 1,553 parameters while retaining at least 99% accuracy, because its removed common vector is output-softmax-invariant and can be transferred exactly into the retained fifth positional common mode.

INTENDED_EDIT: Reparameterize one token-content column with `VOCAB_SIZE - 1` orthogonal coordinates, transfer its initialization common mode into the corresponding positional coordinate, and optimize the new token parameter without weight decay.

EVIDENCE: Removing the fifth positional common mode collapsed accuracy to 38.22%, indicating that its positional optimization pathway is load-bearing; this reverse quotient preserves that pathway while removing the equivalent vocabulary-common token direction instead.

<<<<<<< SEARCH
    """Globally mean-free tied embedding with isolated token-row means."""
=======
    """Tied embedding with one token-common content direction removed."""
>>>>>>> REPLACE

<<<<<<< SEARCH
        row_average = full_weight.mean(dim=1)
        self.weight = nn.Parameter(
            ((full_weight - row_average.unsqueeze(1)) @ content_basis).clone()
        )
        self.row_mean = nn.Parameter(
            (math.sqrt(embedding_dim) * (row_average @ mean_basis)).clone()
        )

    def full_weight(self) -> torch.Tensor:
        centered = self.weight @ self.content_basis.transpose(0, 1)
        row_offsets = (
            (self.mean_basis @ self.row_mean).unsqueeze(1)
            / math.sqrt(self.embedding_dim)
        )
        return centered + row_offsets
=======
        row_average = full_weight.mean(dim=1)
        content_coordinates = (
            (full_weight - row_average.unsqueeze(1)) @ content_basis
        )
        self.weight = nn.Parameter(
            torch.cat(
                (content_coordinates[:, :4], content_coordinates[:, 5:]), dim=1
            ).clone()
        )
        self.fifth_coordinate = nn.Parameter(
            (content_coordinates[:, 4] @ mean_basis).clone()
        )
        self.row_mean = nn.Parameter(
            (math.sqrt(embedding_dim) * (row_average @ mean_basis)).clone()
        )

    def full_weight(self) -> torch.Tensor:
        fifth = self.mean_basis @ self.fifth_coordinate
        content_coordinates = torch.cat(
            (self.weight[:, :4], fifth.unsqueeze(1), self.weight[:, 4:]), dim=1
        )
        centered = content_coordinates @ self.content_basis.transpose(0, 1)
        row_offsets = (
            (self.mean_basis @ self.row_mean).unsqueeze(1)
            / math.sqrt(self.embedding_dim)
        )
        return centered + row_offsets
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.token_emb = MeanFreeTokenEmbedding(cfg.vocab_size, cfg.d_model)
        self.pos_emb = MeanFreePositionEmbedding(cfg.max_seq_len, cfg.d_model)
        self.drop = nn.Dropout(cfg.dropout)
=======
        self.token_emb = MeanFreeTokenEmbedding(cfg.vocab_size, cfg.d_model)
        self.pos_emb = MeanFreePositionEmbedding(cfg.max_seq_len, cfg.d_model)
        # The vocabulary-common fifth content coordinate is a softmax-null
        # output shift; retain its input effect through the positional table.
        object.__setattr__(
            self.token_emb, "position_embedding", self.pos_emb
        )
        self.drop = nn.Dropout(cfg.dropout)
>>>>>>> REPLACE

<<<<<<< SEARCH
                nn.init.normal_(full, mean=0.0, std=0.02)
                full.sub_(full.mean(dim=1, keepdim=True))
                embedding.weight.copy_(full @ embedding.content_basis)
                embedding.row_mean.zero_()
        elif isinstance(module, MeanFreeTokenEmbedding):
            with torch.no_grad():
                full = torch.empty(
                    module.num_embeddings,
                    module.embedding_dim,
                    device=module.weight.device,
                    dtype=module.weight.dtype,
                )
                nn.init.normal_(full, mean=0.0, std=0.02)
                full.sub_(full.mean(dim=1, keepdim=True))
                module.weight.copy_(full @ module.content_basis)
                module.row_mean.zero_()
=======
                nn.init.normal_(full, mean=0.0, std=0.02)
                full.sub_(full.mean(dim=1, keepdim=True))
                content_coordinates = full @ embedding.content_basis
                fifth_common = content_coordinates[:, 4].mean()
                embedding.weight.copy_(
                    torch.cat(
                        (
                            content_coordinates[:, :4],
                            content_coordinates[:, 5:],
                        ),
                        dim=1,
                    )
                )
                embedding.fifth_coordinate.copy_(
                    content_coordinates[:, 4] @ embedding.mean_basis
                )
                embedding.row_mean.zero_()
                embedding.position_embedding.weight[:, 4].add_(fifth_common)
        elif isinstance(module, MeanFreeTokenEmbedding):
            with torch.no_grad():
                full = torch.empty(
                    module.num_embeddings,
                    module.embedding_dim,
                    device=module.weight.device,
                    dtype=module.weight.dtype,
                )
                nn.init.normal_(full, mean=0.0, std=0.02)
                full.sub_(full.mean(dim=1, keepdim=True))
                content_coordinates = full @ module.content_basis
                module.weight.copy_(
                    torch.cat(
                        (
                            content_coordinates[:, :4],
                            content_coordinates[:, 5:],
                        ),
                        dim=1,
                    )
                )
                module.fifth_coordinate.copy_(
                    content_coordinates[:, 4] @ module.mean_basis
                )
                module.row_mean.zero_()
>>>>>>> REPLACE

<<<<<<< SEARCH
    embedding_params = [model.token_emb.weight, model.token_emb.row_mean]
    decay_params = [
        param
        for param in model.parameters()
        if param is not model.token_emb.weight
        and param is not model.token_emb.row_mean
    ]
=======
    embedding_params = [
        model.token_emb.weight,
        model.token_emb.fifth_coordinate,
        model.token_emb.row_mean,
    ]
    decay_params = [
        param
        for param in model.parameters()
        if param is not model.token_emb.weight
        and param is not model.token_emb.fifth_coordinate
        and param is not model.token_emb.row_mean
    ]
>>>>>>> REPLACE