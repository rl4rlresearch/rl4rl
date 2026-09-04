MECHANISM: Second joint token–position common-column gauge

HYPOTHESIS: Fixing feature 0 alongside feature 4 will reduce the model from 1616 to 1615 parameters while retaining at least 99% accuracy, because both offsets are transferred into every positional row and the generalized virtual AdamW path reconstructs and updates all three omitted token coordinates.

INTENDED_EDIT: Generalize the successful token–position embedding quotient from feature 4 to features 0 and 4, including initialization transfer, virtual gradients, clipping, and optimizer projection.

EVIDENCE: The feature-4 quotient achieved 99.93% at 1616 parameters. After feature 1 failed at 66.38%, feature 0 is the strongest remaining coordinate: its LayerNorm-scale gauge achieved 99.83%, and its analogous value-bias gauge previously supported 99.94%.

<<<<<<< SEARCH
        self.transfer_feature = 4
        self.transfer_index = (
            (num_embeddings - 1) * embedding_dim + self.transfer_feature
        )
        self.global_index = num_embeddings * embedding_dim - 1
        self.fixed_indices = (self.transfer_index, self.global_index)

        # Match nn.Embedding's constructor-time random-number consumption.
        full_weight = torch.empty(num_embeddings, embedding_dim)
        nn.init.normal_(full_weight)
        fixed, transfer_offset = self._reduce(full_weight)
        self.weight = nn.Parameter(fixed)
        self.register_buffer(
            "_transfer_offset",
            transfer_offset,
            persistent=False,
        )
=======
        self.transfer_features = (0, 4)
        self.transfer_indices = tuple(
            (num_embeddings - 1) * embedding_dim + feature
            for feature in self.transfer_features
        )
        self.global_index = num_embeddings * embedding_dim - 1
        self.fixed_indices = (*self.transfer_indices, self.global_index)

        # Match nn.Embedding's constructor-time random-number consumption.
        full_weight = torch.empty(num_embeddings, embedding_dim)
        nn.init.normal_(full_weight)
        fixed, transfer_offsets = self._reduce(full_weight)
        self.weight = nn.Parameter(fixed)
        self.register_buffer(
            "_transfer_offsets",
            transfer_offsets,
            persistent=False,
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
        global_anchor = gauged.reshape(-1)[self.global_index].clone()
        gauged.sub_(global_anchor)
        transfer_offset = gauged[-1, self.transfer_feature].clone()
        gauged[:, self.transfer_feature].sub_(transfer_offset)
        flat = gauged.reshape(-1)
        return (
            flat[self._keep_mask(flat.device)].clone(),
            transfer_offset,
        )

    @property
    def transfer_offset(self) -> torch.Tensor:
        return self._transfer_offset
=======
        global_anchor = gauged.reshape(-1)[self.global_index].clone()
        gauged.sub_(global_anchor)
        transfer_offsets = gauged[
            -1, list(self.transfer_features)
        ].clone()
        for feature, offset in zip(
            self.transfer_features,
            transfer_offsets,
        ):
            gauged[:, feature].sub_(offset)
        flat = gauged.reshape(-1)
        return (
            flat[self._keep_mask(flat.device)].clone(),
            transfer_offsets,
        )

    @property
    def transfer_offsets(self) -> torch.Tensor:
        return self._transfer_offsets
>>>>>>> REPLACE

<<<<<<< SEARCH
    def reset_from_full_(self, full_weight: torch.Tensor) -> None:
        fixed, transfer_offset = self._reduce(full_weight)
        self.weight.copy_(fixed)
        self._transfer_offset.copy_(transfer_offset)
=======
    def reset_from_full_(self, full_weight: torch.Tensor) -> None:
        fixed, transfer_offsets = self._reduce(full_weight)
        self.weight.copy_(fixed)
        self._transfer_offsets.copy_(transfer_offsets)
>>>>>>> REPLACE

<<<<<<< SEARCH
        with torch.no_grad():
            self.pos_emb.weight[:, self.token_emb.transfer_feature].add_(
                self.token_emb.transfer_offset
            )
=======
        with torch.no_grad():
            for feature, offset in zip(
                self.token_emb.transfer_features,
                self.token_emb.transfer_offsets,
            ):
                self.pos_emb.weight[:, feature].add_(offset)
>>>>>>> REPLACE

<<<<<<< SEARCH
class TokenPositionGaugeAdamW:
    """AdamW with virtual coordinates for two coupled embedding gauges."""
=======
class TokenPositionGaugeAdamW:
    """AdamW with virtual coordinates for coupled embedding gauges."""
>>>>>>> REPLACE

<<<<<<< SEARCH
        embedding_dim: int,
        transfer_feature: int,
        lr: float,
=======
        embedding_dim: int,
        transfer_features,
        lr: float,
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.embedding_dim = embedding_dim
        self.transfer_feature = transfer_feature
        self.transfer_index = (
            (num_embeddings - 1) * embedding_dim + transfer_feature
        )
        self.global_index = num_embeddings * embedding_dim - 1
        self.fixed_indices = (self.transfer_index, self.global_index)
=======
        self.embedding_dim = embedding_dim
        self.transfer_features = tuple(transfer_features)
        self.transfer_indices = tuple(
            (num_embeddings - 1) * embedding_dim + feature
            for feature in self.transfer_features
        )
        self.global_index = num_embeddings * embedding_dim - 1
        self.fixed_indices = (*self.transfer_indices, self.global_index)
>>>>>>> REPLACE

<<<<<<< SEARCH
        full_token_numel = token_parameter.numel() + 2
=======
        full_token_numel = token_parameter.numel() + len(self.fixed_indices)
>>>>>>> REPLACE

<<<<<<< SEARCH
        keep = torch.ones(
            self.token_parameter.numel() + 2,
            dtype=torch.bool,
            device=self.token_parameter.device,
        )
=======
        keep = torch.ones(
            self.token_parameter.numel() + len(self.fixed_indices),
            dtype=torch.bool,
            device=self.token_parameter.device,
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
        transfer_grad = (
            position_matrix[:, self.transfer_feature].sum()
            - token_matrix[:, self.transfer_feature].sum()
        )
        virtual_token_grad[self.transfer_index] = transfer_grad
        virtual_token_grad[self.global_index] = -virtual_token_grad.sum()
=======
        for feature, transfer_index in zip(
            self.transfer_features,
            self.transfer_indices,
        ):
            virtual_token_grad[transfer_index] = (
                position_matrix[:, feature].sum()
                - token_matrix[:, feature].sum()
            )
        virtual_token_grad[self.global_index] = -virtual_token_grad.sum()
>>>>>>> REPLACE

<<<<<<< SEARCH
        global_direction = token_direction[self.global_index]
        transfer_direction = token_direction[self.transfer_index]
        quotient_token = token_direction.view(
            self.num_embeddings,
            self.embedding_dim,
        ) - global_direction
        quotient_token[:, self.transfer_feature] = (
            token_direction.view(
                self.num_embeddings,
                self.embedding_dim,
            )[:, self.transfer_feature]
            - transfer_direction
        )
        quotient_token = quotient_token.reshape(-1)[keep]

        quotient_position = position_direction.clone()
        quotient_position[:, self.transfer_feature].add_(
            transfer_direction - global_direction
        )
=======
        global_direction = token_direction[self.global_index]
        token_direction_matrix = token_direction.view(
            self.num_embeddings,
            self.embedding_dim,
        )
        quotient_token = token_direction_matrix - global_direction
        quotient_position = position_direction.clone()
        for feature, transfer_index in zip(
            self.transfer_features,
            self.transfer_indices,
        ):
            transfer_direction = token_direction[transfer_index]
            quotient_token[:, feature] = (
                token_direction_matrix[:, feature] - transfer_direction
            )
            quotient_position[:, feature].add_(
                transfer_direction - global_direction
            )
        quotient_token = quotient_token.reshape(-1)[keep]
>>>>>>> REPLACE

<<<<<<< SEARCH
        model.token_emb.embedding_dim,
        model.token_emb.transfer_feature,
    )
=======
        model.token_emb.embedding_dim,
        model.token_emb.transfer_features,
    )
>>>>>>> REPLACE

<<<<<<< SEARCH
        embedding_dim,
        transfer_feature,
    ) = token_position_gauge
=======
        embedding_dim,
        transfer_features,
    ) = token_position_gauge
>>>>>>> REPLACE

<<<<<<< SEARCH
        transfer_index = (
            (num_embeddings - 1) * embedding_dim + transfer_feature
        )
        global_index = num_embeddings * embedding_dim - 1
        fixed_indices = (transfer_index, global_index)
        virtual_token_grad = token_parameter.grad.detach().float().new_zeros(
            token_parameter.numel() + 2
        )
=======
        transfer_indices = tuple(
            (num_embeddings - 1) * embedding_dim + feature
            for feature in transfer_features
        )
        global_index = num_embeddings * embedding_dim - 1
        fixed_indices = (*transfer_indices, global_index)
        virtual_token_grad = token_parameter.grad.detach().float().new_zeros(
            token_parameter.numel() + len(fixed_indices)
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
        virtual_token_grad[transfer_index] = (
            position_matrix[:, transfer_feature].sum()
            - token_matrix[:, transfer_feature].sum()
        )
        virtual_token_grad[global_index] = -virtual_token_grad.sum()
=======
        for feature, transfer_index in zip(
            transfer_features,
            transfer_indices,
        ):
            virtual_token_grad[transfer_index] = (
                position_matrix[:, feature].sum()
                - token_matrix[:, feature].sum()
            )
        virtual_token_grad[global_index] = -virtual_token_grad.sum()
>>>>>>> REPLACE