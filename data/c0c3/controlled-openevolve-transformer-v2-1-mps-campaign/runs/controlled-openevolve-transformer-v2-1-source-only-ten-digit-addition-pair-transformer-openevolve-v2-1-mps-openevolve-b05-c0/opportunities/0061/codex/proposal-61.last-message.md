MECHANISM: Second joint token–position common-column gauge

HYPOTHESIS: Fixing features 1 and 4 of the final token row will reduce the model from 1616 to 1615 parameters while retaining at least 99% accuracy, because both offsets are transferred into positional embeddings and the generalized virtual AdamW path preserves the corresponding full-model updates.

INTENDED_EDIT: Extend the embedding quotient to features 1 and 4, then generalize initialization, gradient reconstruction, clipping, and optimizer projection for three omitted token coordinates.

EVIDENCE: The feature-4 quotient achieved 99.93% at 1616 parameters; feature 1 is the strongest additional candidate because analogous value-bias and LayerNorm-scale gauges achieved 99.98% and 99.81%. The prior feature-1 implementation was unverified rather than accuracy-rejected.

<<<<<<< SEARCH
class GaugeFixedEmbedding(nn.Module):
    """Embedding with global-shift and token-position transfer gauges fixed."""

    def __init__(self, num_embeddings: int, embedding_dim: int):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.transfer_feature = 4
        self.transfer_index = (
            (num_embeddings - 1) * embedding_dim + self.transfer_feature
        )
        self.global_index = num_embeddings * embedding_dim - 1
        self.fixed_indices = (self.transfer_index, self.global_index)
=======
class GaugeFixedEmbedding(nn.Module):
    """Embedding with global-shift and token-position transfer gauges fixed."""

    def __init__(self, num_embeddings: int, embedding_dim: int):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.transfer_features = (1, 4)
        self.transfer_indices = tuple(
            (num_embeddings - 1) * embedding_dim + feature
            for feature in self.transfer_features
        )
        self.global_index = num_embeddings * embedding_dim - 1
        self.fixed_indices = (*self.transfer_indices, self.global_index)
>>>>>>> REPLACE

<<<<<<< SEARCH
        fixed, transfer_offset = self._reduce(full_weight)
        self.weight = nn.Parameter(fixed)
        self.register_buffer(
            "_transfer_offset",
            transfer_offset,
            persistent=False,
        )
=======
        fixed, transfer_offsets = self._reduce(full_weight)
        self.weight = nn.Parameter(fixed)
        self.register_buffer(
            "_transfer_offsets",
            transfer_offsets,
            persistent=False,
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
        gauged = full_weight.clone()
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
        gauged = full_weight.clone()
        global_anchor = gauged.reshape(-1)[self.global_index].clone()
        gauged.sub_(global_anchor)
        transfer_offsets = []
        for feature in self.transfer_features:
            offset = gauged[-1, feature].clone()
            gauged[:, feature].sub_(offset)
            transfer_offsets.append(offset)
        flat = gauged.reshape(-1)
        return (
            flat[self._keep_mask(flat.device)].clone(),
            torch.stack(transfer_offsets),
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

    def __init__(
        self,
        token_parameter,
        position_parameter,
        num_embeddings: int,
        embedding_dim: int,
        transfer_feature: int,
        lr: float,
        weight_decay: float,
        betas=(0.9, 0.999),
        eps: float = 1e-8,
    ):
        self.token_parameter = token_parameter
        self.position_parameter = position_parameter
        self.parameters = [token_parameter, position_parameter]
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.transfer_feature = transfer_feature
        self.transfer_index = (
            (num_embeddings - 1) * embedding_dim + transfer_feature
        )
        self.global_index = num_embeddings * embedding_dim - 1
        self.fixed_indices = (self.transfer_index, self.global_index)
        self.lr = lr
        self.weight_decay = weight_decay
        self.beta1, self.beta2 = betas
        self.eps = eps

        full_token_numel = token_parameter.numel() + 2
=======
class TokenPositionGaugeAdamW:
    """AdamW with virtual coordinates for coupled embedding gauges."""

    def __init__(
        self,
        token_parameter,
        position_parameter,
        num_embeddings: int,
        embedding_dim: int,
        transfer_features: Tuple[int, ...],
        lr: float,
        weight_decay: float,
        betas=(0.9, 0.999),
        eps: float = 1e-8,
    ):
        self.token_parameter = token_parameter
        self.position_parameter = position_parameter
        self.parameters = [token_parameter, position_parameter]
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.transfer_features = tuple(transfer_features)
        self.transfer_indices = tuple(
            (num_embeddings - 1) * embedding_dim + feature
            for feature in self.transfer_features
        )
        self.global_index = num_embeddings * embedding_dim - 1
        self.fixed_indices = (*self.transfer_indices, self.global_index)
        self.lr = lr
        self.weight_decay = weight_decay
        self.beta1, self.beta2 = betas
        self.eps = eps

        full_token_numel = (
            token_parameter.numel() + len(self.fixed_indices)
        )
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
        token_matrix = virtual_token_grad.view(
            self.num_embeddings,
            self.embedding_dim,
        )
        position_matrix = position_grad.view(-1, self.embedding_dim)
        transfer_grad = (
            position_matrix[:, self.transfer_feature].sum()
            - token_matrix[:, self.transfer_feature].sum()
        )
        virtual_token_grad[self.transfer_index] = transfer_grad
        virtual_token_grad[self.global_index] = -virtual_token_grad.sum()
=======
        token_matrix = virtual_token_grad.view(
            self.num_embeddings,
            self.embedding_dim,
        )
        position_matrix = position_grad.view(-1, self.embedding_dim)
        for transfer_feature, transfer_index in zip(
            self.transfer_features,
            self.transfer_indices,
        ):
            virtual_token_grad[transfer_index] = (
                position_matrix[:, transfer_feature].sum()
                - token_matrix[:, transfer_feature].sum()
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
        for transfer_feature, transfer_index in zip(
            self.transfer_features,
            self.transfer_indices,
        ):
            transfer_direction = token_direction[transfer_index]
            quotient_token[:, transfer_feature] = (
                token_direction_matrix[:, transfer_feature]
                - transfer_direction
            )
            quotient_position[:, transfer_feature].add_(
                transfer_direction - global_direction
            )
        quotient_token = quotient_token.reshape(-1)[keep]
>>>>>>> REPLACE

<<<<<<< SEARCH
    # Recover the two omitted token gradients from the coupled invariances.
    (
        token_parameter,
        position_parameter,
        num_embeddings,
        embedding_dim,
        transfer_feature,
    ) = token_position_gauge
    if (
        token_parameter.grad is not None
        and position_parameter.grad is not None
    ):
        transfer_index = (
            (num_embeddings - 1) * embedding_dim + transfer_feature
        )
        global_index = num_embeddings * embedding_dim - 1
        fixed_indices = (transfer_index, global_index)
        virtual_token_grad = token_parameter.grad.detach().float().new_zeros(
            token_parameter.numel() + 2
        )
        keep = torch.ones(
            virtual_token_grad.numel(),
            dtype=torch.bool,
            device=virtual_token_grad.device,
        )
        keep[list(fixed_indices)] = False
        virtual_token_grad[keep] = (
            token_parameter.grad.detach().reshape(-1).float()
        )
        token_matrix = virtual_token_grad.view(
            num_embeddings,
            embedding_dim,
        )
        position_matrix = (
            position_parameter.grad.detach().float().view(-1, embedding_dim)
        )
        virtual_token_grad[transfer_index] = (
            position_matrix[:, transfer_feature].sum()
            - token_matrix[:, transfer_feature].sum()
        )
        virtual_token_grad[global_index] = -virtual_token_grad.sum()
        total_sq.add_(
            virtual_token_grad[list(fixed_indices)].pow(2).sum()
        )
=======
    # Recover the omitted token gradients from the coupled invariances.
    (
        token_parameter,
        position_parameter,
        num_embeddings,
        embedding_dim,
        transfer_features,
    ) = token_position_gauge
    if (
        token_parameter.grad is not None
        and position_parameter.grad is not None
    ):
        transfer_indices = tuple(
            (num_embeddings - 1) * embedding_dim + feature
            for feature in transfer_features
        )
        global_index = num_embeddings * embedding_dim - 1
        fixed_indices = (*transfer_indices, global_index)
        virtual_token_grad = token_parameter.grad.detach().float().new_zeros(
            token_parameter.numel() + len(fixed_indices)
        )
        keep = torch.ones(
            virtual_token_grad.numel(),
            dtype=torch.bool,
            device=virtual_token_grad.device,
        )
        keep[list(fixed_indices)] = False
        virtual_token_grad[keep] = (
            token_parameter.grad.detach().reshape(-1).float()
        )
        token_matrix = virtual_token_grad.view(
            num_embeddings,
            embedding_dim,
        )
        position_matrix = (
            position_parameter.grad.detach().float().view(-1, embedding_dim)
        )
        for transfer_feature, transfer_index in zip(
            transfer_features,
            transfer_indices,
        ):
            virtual_token_grad[transfer_index] = (
                position_matrix[:, transfer_feature].sum()
                - token_matrix[:, transfer_feature].sum()
            )
        virtual_token_grad[global_index] = -virtual_token_grad.sum()
        total_sq.add_(
            virtual_token_grad[list(fixed_indices)].pow(2).sum()
        )
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