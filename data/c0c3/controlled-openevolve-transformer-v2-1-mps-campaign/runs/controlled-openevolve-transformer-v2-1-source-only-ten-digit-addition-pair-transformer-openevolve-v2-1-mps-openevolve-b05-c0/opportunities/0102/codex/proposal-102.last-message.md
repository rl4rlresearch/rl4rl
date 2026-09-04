MECHANISM: Virtual-AdamW positional common-shift quotient

HYPOTHESIS: Gauge-fixing one uniform scalar direction of the positional embedding will reduce the verified model from 1604 to 1603 parameters while retaining at least 99% accuracy, because the shift is exactly removed by the pre-norm/final-LayerNorm computation and full-coordinate AdamW moments are preserved virtually.

INTENDED_EDIT: Replace the positional embedding with a one-coordinate gauge-fixed embedding and extend the coupled token-position optimizer and gradient clipping to reconstruct and optimize its omitted coordinate.

EVIDENCE: The 1604-parameter design achieved 99.92% accuracy using an analogous global-shift quotient in the token embedding and virtual-AdamW common-shift quotients in learned projections; this applies the same proven invariance to the previously untouched positional table without adding another attention-output quotient, whose fourth-coordinate trials timed out.

<<<<<<< SEARCH
    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        return F.embedding(idx, self.full_weight())


class GaugeFixedQKV(nn.Module):
=======
    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        return F.embedding(idx, self.full_weight())


class GaugeFixedPositionEmbedding(nn.Module):
    """Position embedding with one uniform scalar direction fixed."""

    def __init__(self, num_embeddings: int, embedding_dim: int):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.fixed_index = num_embeddings * embedding_dim - 1

        # Match nn.Embedding's constructor-time random-number consumption.
        source = nn.Embedding(num_embeddings, embedding_dim)
        self.weight = nn.Parameter(self._reduce(source.weight.detach()))

    def _reduce(self, full_weight: torch.Tensor) -> torch.Tensor:
        gauged = full_weight.clone()
        anchor = gauged.reshape(-1)[self.fixed_index].clone()
        gauged.sub_(anchor)
        return gauged.reshape(-1)[:-1].clone()

    def full_weight(self) -> torch.Tensor:
        flat = torch.cat((self.weight, self.weight.new_zeros(1)))
        return flat.view(self.num_embeddings, self.embedding_dim)

    @torch.no_grad()
    def reset_from_full_(self, full_weight: torch.Tensor) -> None:
        self.weight.copy_(self._reduce(full_weight))

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        return F.embedding(idx, self.full_weight())


class GaugeFixedQKV(nn.Module):
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.pos_emb = nn.Embedding(cfg.max_seq_len, cfg.d_model)
=======
        self.pos_emb = GaugeFixedPositionEmbedding(
            cfg.max_seq_len,
            cfg.d_model,
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
            nn.init.normal_(full_weight, mean=0.0, std=0.02)
            module.reset_from_full_(full_weight)
        elif isinstance(module, GaugeFixedQKV):
=======
            nn.init.normal_(full_weight, mean=0.0, std=0.02)
            module.reset_from_full_(full_weight)
        elif isinstance(module, GaugeFixedPositionEmbedding):
            full_weight = torch.empty(
                module.num_embeddings,
                module.embedding_dim,
                device=module.weight.device,
                dtype=module.weight.dtype,
            )
            nn.init.normal_(full_weight, mean=0.0, std=0.02)
            module.reset_from_full_(full_weight)
        elif isinstance(module, GaugeFixedQKV):
>>>>>>> REPLACE

<<<<<<< SEARCH
        with torch.no_grad():
            self.pos_emb.weight[:, self.token_emb.transfer_feature].add_(
                self.token_emb.transfer_offset
            )
=======
        with torch.no_grad():
            full_position_weight = self.pos_emb.full_weight()
            full_position_weight[:, self.token_emb.transfer_feature].add_(
                self.token_emb.transfer_offset
            )
            self.pos_emb.reset_from_full_(full_position_weight)
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
=======
class TokenPositionGaugeAdamW:
    """AdamW with virtual coordinates for three embedding gauges."""

    def __init__(
        self,
        token_parameter,
        position_parameter,
        num_embeddings: int,
        embedding_dim: int,
        transfer_feature: int,
        position_embeddings: int,
        lr: float,
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.transfer_feature = transfer_feature
        self.transfer_index = (
            (num_embeddings - 1) * embedding_dim + transfer_feature
        )
        self.global_index = num_embeddings * embedding_dim - 1
        self.fixed_indices = (self.transfer_index, self.global_index)
        self.lr = lr
=======
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.transfer_feature = transfer_feature
        self.position_embeddings = position_embeddings
        self.transfer_index = (
            (num_embeddings - 1) * embedding_dim + transfer_feature
        )
        self.global_index = num_embeddings * embedding_dim - 1
        self.fixed_indices = (self.transfer_index, self.global_index)
        self.position_global_index = (
            position_embeddings * embedding_dim - 1
        )
        self.lr = lr
>>>>>>> REPLACE

<<<<<<< SEARCH
            "position_exp_avg": torch.zeros_like(position_parameter),
            "position_exp_avg_sq": torch.zeros_like(position_parameter),
=======
            "position_exp_avg": torch.zeros(
                position_parameter.numel() + 1,
                device=position_parameter.device,
                dtype=position_parameter.dtype,
            ),
            "position_exp_avg_sq": torch.zeros(
                position_parameter.numel() + 1,
                device=position_parameter.device,
                dtype=position_parameter.dtype,
            ),
>>>>>>> REPLACE

<<<<<<< SEARCH
        keep[list(self.fixed_indices)] = False
        return keep

    def zero_grad(self, set_to_none: bool = True) -> None:
=======
        keep[list(self.fixed_indices)] = False
        return keep

    def _position_keep_mask(self) -> torch.Tensor:
        keep = torch.ones(
            self.position_parameter.numel() + 1,
            dtype=torch.bool,
            device=self.position_parameter.device,
        )
        keep[self.position_global_index] = False
        return keep

    def zero_grad(self, set_to_none: bool = True) -> None:
>>>>>>> REPLACE

<<<<<<< SEARCH
        keep = self._keep_mask()
        virtual_token_grad = self.token_parameter.grad.new_zeros(keep.numel())
        virtual_token_grad[keep] = (
            self.token_parameter.grad.detach().reshape(-1)
        )
        position_grad = self.position_parameter.grad.detach()

        token_matrix = virtual_token_grad.view(
            self.num_embeddings,
            self.embedding_dim,
        )
        position_matrix = position_grad.view(-1, self.embedding_dim)
        transfer_grad = (
            position_matrix[:, self.transfer_feature].sum()
            - token_matrix[:, self.transfer_feature].sum()
        )
=======
        keep = self._keep_mask()
        virtual_token_grad = self.token_parameter.grad.new_zeros(keep.numel())
        virtual_token_grad[keep] = (
            self.token_parameter.grad.detach().reshape(-1)
        )

        position_keep = self._position_keep_mask()
        virtual_position_grad = self.position_parameter.grad.new_zeros(
            position_keep.numel()
        )
        virtual_position_grad[position_keep] = (
            self.position_parameter.grad.detach().reshape(-1)
        )
        virtual_position_grad[self.position_global_index] = (
            -virtual_position_grad.sum()
        )

        token_matrix = virtual_token_grad.view(
            self.num_embeddings,
            self.embedding_dim,
        )
        position_matrix = virtual_position_grad.view(
            self.position_embeddings,
            self.embedding_dim,
        )
        transfer_grad = (
            position_matrix[:, self.transfer_feature].sum()
            - token_matrix[:, self.transfer_feature].sum()
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
        position_exp_avg.mul_(self.beta1).add_(
            position_grad,
            alpha=1.0 - self.beta1,
        )
        position_exp_avg_sq.mul_(self.beta2).addcmul_(
            position_grad,
            position_grad,
            value=1.0 - self.beta2,
        )
=======
        position_exp_avg.mul_(self.beta1).add_(
            virtual_position_grad,
            alpha=1.0 - self.beta1,
        )
        position_exp_avg_sq.mul_(self.beta2).addcmul_(
            virtual_position_grad,
            virtual_position_grad,
            value=1.0 - self.beta2,
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
        quotient_position = position_direction.clone()
        quotient_position[:, self.transfer_feature].add_(
            transfer_direction - global_direction
        )

        self.token_parameter.mul_(1.0 - self.lr * self.weight_decay)
=======
        position_global_direction = position_direction[
            self.position_global_index
        ]
        quotient_position_full = position_direction.view(
            self.position_embeddings,
            self.embedding_dim,
        ) - position_global_direction
        quotient_position_full[:, self.transfer_feature].add_(
            transfer_direction - global_direction
        )
        quotient_position = quotient_position_full.reshape(-1)[position_keep]

        self.token_parameter.mul_(1.0 - self.lr * self.weight_decay)
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
        model.token_emb.transfer_feature,
        model.pos_emb.num_embeddings,
    )
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
    # Recover the omitted position and token gradients from their invariances.
    (
        token_parameter,
        position_parameter,
        num_embeddings,
        embedding_dim,
        transfer_feature,
        position_embeddings,
    ) = token_position_gauge
    if (
        token_parameter.grad is not None
        and position_parameter.grad is not None
    ):
        position_global_index = (
            position_embeddings * embedding_dim - 1
        )
        position_keep = torch.ones(
            position_parameter.numel() + 1,
            dtype=torch.bool,
            device=position_parameter.grad.device,
        )
        position_keep[position_global_index] = False
        virtual_position_grad = (
            position_parameter.grad.detach().float().new_zeros(
                position_parameter.numel() + 1
            )
        )
        virtual_position_grad[position_keep] = (
            position_parameter.grad.detach().reshape(-1).float()
        )
        virtual_position_grad[position_global_index] = (
            -virtual_position_grad.sum()
        )

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
        position_matrix = virtual_position_grad.view(
            position_embeddings,
            embedding_dim,
        )
        virtual_token_grad[transfer_index] = (
            position_matrix[:, transfer_feature].sum()
            - token_matrix[:, transfer_feature].sum()
        )
        virtual_token_grad[global_index] = -virtual_token_grad.sum()
        total_sq.add_(
            virtual_token_grad[list(fixed_indices)].pow(2).sum()
        )
        total_sq.add_(
            virtual_position_grad[position_global_index].pow(2)
        )
>>>>>>> REPLACE