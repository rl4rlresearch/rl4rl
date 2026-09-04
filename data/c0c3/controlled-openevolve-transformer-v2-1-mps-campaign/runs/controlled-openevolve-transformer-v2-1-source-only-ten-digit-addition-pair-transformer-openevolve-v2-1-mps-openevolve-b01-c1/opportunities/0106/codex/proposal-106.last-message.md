MECHANISM: Token-position common-mode quotient

HYPOTHESIS: Centering one token-content coordinate across the vocabulary and transferring its common component to the matching position coordinate will reduce the model from 1,534 to 1,533 parameters while retaining at least 99% accuracy, because token-plus-position inputs are preserved and tied output logits change only by a softmax-null vocabulary-common scalar.

INTENDED_EDIT: Represent the first token-content coordinate with a vocabulary mean-free basis, transfer its removed mean into position embeddings after fresh initialization, and keep all token-embedding coordinates exempt from weight decay.

EVIDENCE: The 1,534-parameter design reached 99.91%, while rank-five token factorization fell to 90.42% after removing many token-relative degrees. This tests a single exact embedding gauge instead, after every candidate fifth `ln1` bias coordinate failed.

<<<<<<< SEARCH
        row_average = full_weight.mean(dim=1)
        self.weight = nn.Parameter(
            ((full_weight - row_average.unsqueeze(1)) @ content_basis).clone()
        )
        self.row_mean = nn.Parameter(
            (math.sqrt(embedding_dim) * (row_average @ mean_basis)).clone()
        )
=======
        row_average = full_weight.mean(dim=1)
        content_coordinates = (
            (full_weight - row_average.unsqueeze(1)) @ content_basis
        )
        first_coordinate = content_coordinates[:, 0]
        self.first_coordinate = nn.Parameter(
            (first_coordinate @ mean_basis).clone()
        )
        self.weight = nn.Parameter(content_coordinates[:, 1:].clone())
        self.register_buffer(
            "common_coordinate",
            first_coordinate.mean().clone(),
            persistent=False,
        )
        self.row_mean = nn.Parameter(
            (math.sqrt(embedding_dim) * (row_average @ mean_basis)).clone()
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
    def full_weight(self) -> torch.Tensor:
        centered = self.weight @ self.content_basis.transpose(0, 1)
        row_offsets = (
=======
    def full_weight(self) -> torch.Tensor:
        first_coordinate = (
            self.mean_basis @ self.first_coordinate
        ).unsqueeze(1)
        content_coordinates = torch.cat(
            (first_coordinate, self.weight), dim=1
        )
        centered = content_coordinates @ self.content_basis.transpose(0, 1)
        row_offsets = (
>>>>>>> REPLACE

<<<<<<< SEARCH
                full.sub_(full.mean(dim=1, keepdim=True))
                embedding.weight.copy_(full @ embedding.content_basis)
                embedding.row_mean.zero_()
=======
                full.sub_(full.mean(dim=1, keepdim=True))
                content_coordinates = full @ embedding.content_basis
                first_coordinate = content_coordinates[:, 0]
                embedding.first_coordinate.copy_(
                    first_coordinate @ embedding.mean_basis
                )
                embedding.weight.copy_(content_coordinates[:, 1:])
                embedding.common_coordinate.copy_(first_coordinate.mean())
                embedding.row_mean.zero_()
>>>>>>> REPLACE

<<<<<<< SEARCH
                full.sub_(full.mean(dim=1, keepdim=True))
                module.weight.copy_(full @ module.content_basis)
                module.row_mean.zero_()
=======
                full.sub_(full.mean(dim=1, keepdim=True))
                content_coordinates = full @ module.content_basis
                first_coordinate = content_coordinates[:, 0]
                module.first_coordinate.copy_(
                    first_coordinate @ module.mean_basis
                )
                module.weight.copy_(content_coordinates[:, 1:])
                module.common_coordinate.copy_(first_coordinate.mean())
                module.row_mean.zero_()
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.apply(self._init_weights)

    @staticmethod
=======
        self.apply(self._init_weights)

        # Move the removed vocabulary-common token component into positions.
        # This preserves every initial token-plus-position input, while its
        # removal from the tied output adds only a common logit offset.
        with torch.no_grad():
            self.pos_emb.weight[:, 0].add_(
                self.token_emb.common_coordinate
            )

    @staticmethod
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
        model.token_emb.first_coordinate,
        model.token_emb.row_mean,
    ]
    decay_params = [
        param
        for param in model.parameters()
        if param is not model.token_emb.weight
        and param is not model.token_emb.first_coordinate
        and param is not model.token_emb.row_mean
    ]
>>>>>>> REPLACE