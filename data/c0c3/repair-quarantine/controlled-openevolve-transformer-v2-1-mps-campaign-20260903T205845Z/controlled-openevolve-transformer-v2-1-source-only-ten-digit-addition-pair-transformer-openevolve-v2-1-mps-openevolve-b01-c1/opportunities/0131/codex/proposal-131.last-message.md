MECHANISM: Reverse token–position common-mode quotient

HYPOTHESIS: Removing the vocabulary-common component of the first token-content coordinate will reduce the model from 1,525 to 1,524 parameters while retaining at least 99% accuracy, because the combined input offset remains trainable through the load-bearing positional common mode and the removed output component is softmax-null.

INTENDED_EDIT: Represent the first token-content coordinate in a vocabulary-mean-free basis, transfer its initialized common component to positional coordinate zero, and preserve no-weight-decay treatment for all token-embedding parameters.

EVIDENCE: Transferring positional coordinate zero’s common component into token embeddings reduced accuracy to 93.88%, indicating that the positional parameterization is optimization-important. This patch reverses that quotient: it preserves the positional common degree of freedom from the 99.88%-accurate design while removing the mathematically redundant token-side common component.

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
=======
        row_average = full_weight.mean(dim=1)
        projected = (
            (full_weight - row_average.unsqueeze(1)) @ content_basis
        )
        self.first_coordinate = nn.Parameter(
            (projected[:, 0] @ mean_basis).clone()
        )
        self.weight = nn.Parameter(projected[:, 1:].clone())
        self.row_mean = nn.Parameter(
            (math.sqrt(embedding_dim) * (row_average @ mean_basis)).clone()
        )
        self.register_buffer(
            "removed_first_common",
            projected[:, 0].mean().clone(),
            persistent=False,
        )

    def full_weight(self) -> torch.Tensor:
        first_coordinate = (
            self.mean_basis @ self.first_coordinate
        ).unsqueeze(1)
        content_coordinates = torch.cat(
            (first_coordinate, self.weight), dim=1
        )
        centered = (
            content_coordinates @ self.content_basis.transpose(0, 1)
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
        # Preserve the initialized token-plus-position inputs after removing the
        # second, third, and fourth positional common modes. The corresponding
        # token shifts are also softmax-null in the tied output projection.
        with torch.no_grad():
            self.token_emb.weight[:, 1].add_(
                self.pos_emb.removed_second_common
            )
            self.token_emb.weight[:, 2].add_(
                self.pos_emb.removed_third_common
            )
            self.token_emb.weight[:, 3].add_(
                self.pos_emb.removed_fourth_common
            )
            self.token_emb.weight[:, 4].add_(
                self.pos_emb.removed_fifth_common
            )
=======
        # Preserve the initialized token-plus-position inputs while assigning
        # their first common mode to positions and the next four to tokens.
        # Every removed token-output component is softmax-null.
        with torch.no_grad():
            self.pos_emb.weight[:, 0].add_(
                self.token_emb.removed_first_common
            )
            self.token_emb.weight[:, 0].add_(
                self.pos_emb.removed_second_common
            )
            self.token_emb.weight[:, 1].add_(
                self.pos_emb.removed_third_common
            )
            self.token_emb.weight[:, 2].add_(
                self.pos_emb.removed_fourth_common
            )
            self.token_emb.weight[:, 3].add_(
                self.pos_emb.removed_fifth_common
            )
>>>>>>> REPLACE

<<<<<<< SEARCH
                full.sub_(full.mean(dim=1, keepdim=True))
                embedding.weight.copy_(full @ embedding.content_basis)
                embedding.row_mean.zero_()
=======
                full.sub_(full.mean(dim=1, keepdim=True))
                projected = full @ embedding.content_basis
                embedding.first_coordinate.copy_(
                    projected[:, 0] @ embedding.mean_basis
                )
                embedding.weight.copy_(projected[:, 1:])
                embedding.removed_first_common.copy_(
                    projected[:, 0].mean()
                )
                embedding.row_mean.zero_()
>>>>>>> REPLACE

<<<<<<< SEARCH
                full.sub_(full.mean(dim=1, keepdim=True))
                module.weight.copy_(full @ module.content_basis)
                module.row_mean.zero_()
=======
                full.sub_(full.mean(dim=1, keepdim=True))
                projected = full @ module.content_basis
                module.first_coordinate.copy_(
                    projected[:, 0] @ module.mean_basis
                )
                module.weight.copy_(projected[:, 1:])
                module.removed_first_common.copy_(
                    projected[:, 0].mean()
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