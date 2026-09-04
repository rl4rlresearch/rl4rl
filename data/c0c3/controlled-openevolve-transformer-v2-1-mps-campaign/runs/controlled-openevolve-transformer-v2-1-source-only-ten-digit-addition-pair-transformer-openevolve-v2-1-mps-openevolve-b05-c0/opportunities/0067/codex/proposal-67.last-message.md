MECHANISM: Exact latent full-embedding AdamW under a second token–position gauge

HYPOTHESIS: Fixing feature 3 alongside feature 4 will reduce the model from 1616 to 1615 parameters while retaining at least 99% accuracy because maintaining and updating gauge-equivalent full token and positional tensors reproduces full-parameter AdamW exactly, avoiding the projected optimizer dynamics of the 95.46% feature-3 attempt.

INTENDED_EDIT: Remove the final-token feature-3 coordinate, transfer its offset into positional embeddings, and replace projected token–position updates with latent full-tensor AdamW followed by exact gauge reduction.

EVIDENCE: The existing feature-4 quotient achieved 99.93%, while feature 3 was the strongest attempted second feature at 95.46%; its much smaller deficit than other second-feature attempts makes optimizer fidelity on that exact quotient the most informative next test.

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

    def _keep_mask(self, device: torch.device) -> torch.Tensor:
        keep = torch.ones(
            self.num_embeddings * self.embedding_dim,
            dtype=torch.bool,
            device=device,
        )
        keep[list(self.fixed_indices)] = False
        return keep

    def _reduce(
        self,
        full_weight: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
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

    def full_weight(self) -> torch.Tensor:
        keep = self._keep_mask(self.weight.device)
        flat = self.weight.new_zeros(keep.numel())
        flat = flat.masked_scatter(keep, self.weight)
        return flat.view(self.num_embeddings, self.embedding_dim)

    @torch.no_grad()
    def reset_from_full_(self, full_weight: torch.Tensor) -> None:
        fixed, transfer_offset = self._reduce(full_weight)
        self.weight.copy_(fixed)
        self._transfer_offset.copy_(transfer_offset)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        return F.embedding(idx, self.full_weight())
=======
class GaugeFixedEmbedding(nn.Module):
    """Embedding with global-shift and two token-position gauges fixed."""

    def __init__(self, num_embeddings: int, embedding_dim: int):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.transfer_features = (3, 4)
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

    def _keep_mask(self, device: torch.device) -> torch.Tensor:
        keep = torch.ones(
            self.num_embeddings * self.embedding_dim,
            dtype=torch.bool,
            device=device,
        )
        keep[list(self.fixed_indices)] = False
        return keep

    def _reduce(
        self,
        full_weight: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        gauged = full_weight.clone()
        global_anchor = gauged.reshape(-1)[self.global_index].clone()
        gauged.sub_(global_anchor)
        transfer_offsets = []
        for feature in self.transfer_features:
            offset = gauged[-1, feature].clone()
            transfer_offsets.append(offset)
            gauged[:, feature].sub_(offset)
        flat = gauged.reshape(-1)
        return (
            flat[self._keep_mask(flat.device)].clone(),
            torch.stack(transfer_offsets),
        )

    @property
    def transfer_offsets(self) -> torch.Tensor:
        return self._transfer_offsets

    def full_weight(self) -> torch.Tensor:
        keep = self._keep_mask(self.weight.device)
        flat = self.weight.new_zeros(keep.numel())
        flat = flat.masked_scatter(keep, self.weight)
        return flat.view(self.num_embeddings, self.embedding_dim)

    @torch.no_grad()
    def reset_from_full_(self, full_weight: torch.Tensor) -> None:
        fixed, transfer_offsets = self._reduce(full_weight)
        self.weight.copy_(fixed)
        self._transfer_offsets.copy_(transfer_offsets)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        return F.embedding(idx, self.full_weight())
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
        self.state = {
            "step": 0,
            "token_exp_avg": torch.zeros(
                full_token_numel,
                device=token_parameter.device,
                dtype=token_parameter.dtype,
            ),
            "token_exp_avg_sq": torch.zeros(
                full_token_numel,
                device=token_parameter.device,
                dtype=token_parameter.dtype,
            ),
            "position_exp_avg": torch.zeros_like(position_parameter),
            "position_exp_avg_sq": torch.zeros_like(position_parameter),
        }
=======
class TokenPositionGaugeAdamW:
    """Full-tensor AdamW followed by exact embedding-gauge reduction."""

    def __init__(
        self,
        token_parameter,
        position_parameter,
        num_embeddings: int,
        embedding_dim: int,
        transfer_features,
        transfer_offsets,
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

        full_token_numel = token_parameter.numel() + len(self.fixed_indices)
        keep = torch.ones(
            full_token_numel,
            dtype=torch.bool,
            device=token_parameter.device,
        )
        keep[list(self.fixed_indices)] = False
        token_value = token_parameter.new_zeros(
            num_embeddings,
            embedding_dim,
        )
        token_value.reshape(-1)[keep] = token_parameter.detach().reshape(-1)
        position_value = position_parameter.detach().clone()
        for feature, offset in zip(
            self.transfer_features,
            transfer_offsets.detach(),
        ):
            token_value[:, feature].add_(offset)
            position_value[:, feature].sub_(offset)

        self.state = {
            "step": 0,
            "token_value": token_value,
            "position_value": position_value,
            "token_exp_avg": torch.zeros(
                full_token_numel,
                device=token_parameter.device,
                dtype=token_parameter.dtype,
            ),
            "token_exp_avg_sq": torch.zeros(
                full_token_numel,
                device=token_parameter.device,
                dtype=token_parameter.dtype,
            ),
            "position_exp_avg": torch.zeros_like(position_parameter),
            "position_exp_avg_sq": torch.zeros_like(position_parameter),
        }
>>>>>>> REPLACE

<<<<<<< SEARCH
    def _keep_mask(self) -> torch.Tensor:
        keep = torch.ones(
            self.token_parameter.numel() + 2,
            dtype=torch.bool,
            device=self.token_parameter.device,
        )
        keep[list(self.fixed_indices)] = False
        return keep
=======
    def _keep_mask(self) -> torch.Tensor:
        keep = torch.ones(
            self.token_parameter.numel() + len(self.fixed_indices),
            dtype=torch.bool,
            device=self.token_parameter.device,
        )
        keep[list(self.fixed_indices)] = False
        return keep
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

        self.token_parameter.mul_(1.0 - self.lr * self.weight_decay)
        self.position_parameter.mul_(1.0 - self.lr * self.weight_decay)
        self.token_parameter.add_(
            quotient_token.view_as(self.token_parameter),
            alpha=-self.lr / bias_correction1,
        )
        self.position_parameter.add_(
            quotient_position.view_as(self.position_parameter),
            alpha=-self.lr / bias_correction1,
        )
=======
        token_value = state["token_value"]
        position_value = state["position_value"]
        token_value.mul_(1.0 - self.lr * self.weight_decay)
        position_value.mul_(1.0 - self.lr * self.weight_decay)
        token_value.add_(
            token_direction.view_as(token_value),
            alpha=-self.lr / bias_correction1,
        )
        position_value.add_(
            position_direction,
            alpha=-self.lr / bias_correction1,
        )

        canonical_token = token_value.clone()
        global_anchor = canonical_token.reshape(-1)[
            self.global_index
        ].clone()
        canonical_token.sub_(global_anchor)
        transfer_offsets = []
        for feature in self.transfer_features:
            offset = canonical_token[-1, feature].clone()
            transfer_offsets.append(offset)
            canonical_token[:, feature].sub_(offset)

        self.token_parameter.copy_(canonical_token.reshape(-1)[keep])
        self.position_parameter.copy_(position_value)
        for feature, offset in zip(
            self.transfer_features,
            transfer_offsets,
        ):
            self.position_parameter[:, feature].add_(offset)
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
    # Recover all omitted token gradients from the coupled invariances.
    (
        token_parameter,
        position_parameter,
        num_embeddings,
        embedding_dim,
        transfer_features,
        _transfer_offsets,
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
        for feature, transfer_index in zip(
            transfer_features,
            transfer_indices,
        ):
            virtual_token_grad[transfer_index] = (
                position_matrix[:, feature].sum()
                - token_matrix[:, feature].sum()
            )
        virtual_token_grad[global_index] = -virtual_token_grad.sum()
        total_sq.add_(
            virtual_token_grad[list(fixed_indices)].pow(2).sum()
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
    token_position_gauge = (
        model.token_emb.weight,
        model.pos_emb.weight,
        model.token_emb.num_embeddings,
        model.token_emb.embedding_dim,
        model.token_emb.transfer_feature,
    )
=======
    token_position_gauge = (
        model.token_emb.weight,
        model.pos_emb.weight,
        model.token_emb.num_embeddings,
        model.token_emb.embedding_dim,
        model.token_emb.transfer_features,
        model.token_emb.transfer_offsets,
    )
>>>>>>> REPLACE