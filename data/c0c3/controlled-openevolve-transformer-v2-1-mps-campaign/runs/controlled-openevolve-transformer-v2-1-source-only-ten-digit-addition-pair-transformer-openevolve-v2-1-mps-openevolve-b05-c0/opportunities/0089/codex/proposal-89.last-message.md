MECHANISM: Virtual-AdamW quotient of a per-position residual-stream shift

HYPOTHESIS: Gauge-fixing positional embedding coordinate `[0, 7]` will reduce the verified model from 1607 to 1606 parameters while retaining at least 99% accuracy, because a uniform feature shift at position 0 is removed by every pre-norm and the final LayerNorm, while virtual gradients and full-coordinate AdamW moments preserve training dynamics.

INTENDED_EDIT: Replace the positional embedding with a one-coordinate gauge-fixed embedding, reconstruct its omitted gradient, and extend the coupled token-position optimizer and clipping logic to project both the existing transfer gauges and the positional row-shift gauge.

EVIDENCE: The 1607-parameter design achieved 99.97% with virtual-AdamW gauge quotients, including a token-position embedding gauge; the prior positional-gauge submission was not evaluated because its SEARCH text matched multiple times, so this distinct exact symmetry remains the most informative untested reduction.

<<<<<<< SEARCH
    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        return F.embedding(idx, self.full_weight())


class GaugeFixedQKV(nn.Module):
=======
    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        return F.embedding(idx, self.full_weight())


class GaugeFixedPositionEmbedding(nn.Module):
    """Position embedding with one residual-stream shift coordinate fixed."""

    def __init__(self, num_embeddings: int, embedding_dim: int):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.fixed_row = 0
        self.fixed_feature = embedding_dim - 1
        self.fixed_index = (
            self.fixed_row * embedding_dim + self.fixed_feature
        )

        # Match nn.Embedding's constructor-time random-number consumption.
        source = nn.Embedding(num_embeddings, embedding_dim)
        self.weight = nn.Parameter(self._reduce(source.weight.detach()))

    def _keep_mask(self, device: torch.device) -> torch.Tensor:
        keep = torch.ones(
            self.num_embeddings * self.embedding_dim,
            dtype=torch.bool,
            device=device,
        )
        keep[self.fixed_index] = False
        return keep

    def _reduce(self, full_weight: torch.Tensor) -> torch.Tensor:
        gauged = full_weight.clone()
        anchor = gauged[self.fixed_row, self.fixed_feature].clone()
        gauged[self.fixed_row].sub_(anchor)
        flat = gauged.reshape(-1)
        return flat[self._keep_mask(flat.device)].clone()

    def full_weight(self) -> torch.Tensor:
        keep = self._keep_mask(self.weight.device)
        flat = self.weight.new_zeros(keep.numel())
        flat = flat.masked_scatter(keep, self.weight)
        return flat.view(self.num_embeddings, self.embedding_dim)

    @torch.no_grad()
    def reset_from_full_(self, full_weight: torch.Tensor) -> None:
        self.weight.copy_(self._reduce(full_weight))

    @torch.no_grad()
    def add_feature_offset_(
        self,
        feature_index: int,
        offset: torch.Tensor,
    ) -> None:
        full_weight = self.full_weight()
        full_weight[:, feature_index].add_(offset)
        self.reset_from_full_(full_weight)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        return F.embedding(idx, self.full_weight())


class GaugeFixedQKV(nn.Module):
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.token_emb = GaugeFixedEmbedding(cfg.vocab_size, cfg.d_model)
        self.pos_emb = nn.Embedding(cfg.max_seq_len, cfg.d_model)
=======
        self.token_emb = GaugeFixedEmbedding(cfg.vocab_size, cfg.d_model)
        self.pos_emb = GaugeFixedPositionEmbedding(
            cfg.max_seq_len,
            cfg.d_model,
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
        if isinstance(module, GaugeFixedEmbedding):
            full_weight = torch.empty(
                module.num_embeddings,
                module.embedding_dim,
                device=module.weight.device,
                dtype=module.weight.dtype,
            )
            nn.init.normal_(full_weight, mean=0.0, std=0.02)
            module.reset_from_full_(full_weight)
        elif isinstance(module, GaugeFixedQKV):
=======
        if isinstance(module, GaugeFixedEmbedding):
            full_weight = torch.empty(
                module.num_embeddings,
                module.embedding_dim,
                device=module.weight.device,
                dtype=module.weight.dtype,
            )
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
            self.pos_emb.add_feature_offset_(
                self.token_emb.transfer_feature,
                self.token_emb.transfer_offset,
            )
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
    """AdamW with virtual coordinates for coupled embedding gauges."""

    def __init__(
        self,
        token_parameter,
        position_parameter,
        num_embeddings: int,
        embedding_dim: int,
        transfer_feature: int,
        position_num_embeddings: int,
        position_fixed_index: int,
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
        self.position_num_embeddings = position_num_embeddings
        self.position_fixed_index = position_fixed_index
        self.position_fixed_row = position_fixed_index // embedding_dim
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
        full_position_numel = position_parameter.numel() + 1
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
            "position_exp_avg": torch.zeros(
                full_position_numel,
                device=position_parameter.device,
                dtype=position_parameter.dtype,
            ),
            "position_exp_avg_sq": torch.zeros(
                full_position_numel,
                device=position_parameter.device,
                dtype=position_parameter.dtype,
            ),
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

    def zero_grad(self, set_to_none: bool = True) -> None:
=======
    def _keep_mask(self) -> torch.Tensor:
        keep = torch.ones(
            self.token_parameter.numel() + 2,
            dtype=torch.bool,
            device=self.token_parameter.device,
        )
        keep[list(self.fixed_indices)] = False
        return keep

    def _position_keep_mask(self) -> torch.Tensor:
        keep = torch.ones(
            self.position_parameter.numel() + 1,
            dtype=torch.bool,
            device=self.position_parameter.device,
        )
        keep[self.position_fixed_index] = False
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
        virtual_token_grad[self.transfer_index] = transfer_grad
        virtual_token_grad[self.global_index] = -virtual_token_grad.sum()
=======
        keep = self._keep_mask()
        position_keep = self._position_keep_mask()
        virtual_token_grad = self.token_parameter.grad.new_zeros(keep.numel())
        virtual_token_grad[keep] = (
            self.token_parameter.grad.detach().reshape(-1)
        )
        virtual_position_grad = self.position_parameter.grad.new_zeros(
            position_keep.numel()
        )
        virtual_position_grad[position_keep] = (
            self.position_parameter.grad.detach().reshape(-1)
        )
        position_row_start = self.position_fixed_row * self.embedding_dim
        position_row_end = position_row_start + self.embedding_dim
        virtual_position_grad[self.position_fixed_index] = (
            -virtual_position_grad[
                position_row_start:position_row_end
            ].sum()
        )

        token_matrix = virtual_token_grad.view(
            self.num_embeddings,
            self.embedding_dim,
        )
        position_matrix = virtual_position_grad.view(
            self.position_num_embeddings,
            self.embedding_dim,
        )
        transfer_grad = (
            position_matrix[:, self.transfer_feature].sum()
            - token_matrix[:, self.transfer_feature].sum()
        )
        virtual_token_grad[self.transfer_index] = transfer_grad
        virtual_token_grad[self.global_index] = -virtual_token_grad.sum()
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
        quotient_position = position_direction.view(
            self.position_num_embeddings,
            self.embedding_dim,
        ).clone()
        quotient_position[:, self.transfer_feature].add_(
            transfer_direction - global_direction
        )
        position_anchor_direction = quotient_position.reshape(-1)[
            self.position_fixed_index
        ].clone()
        quotient_position[self.position_fixed_row].sub_(
            position_anchor_direction
        )
        quotient_position = quotient_position.reshape(-1)[position_keep]

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
    # Recover the omitted token and position gradients from the invariances.
    (
        token_parameter,
        position_parameter,
        num_embeddings,
        embedding_dim,
        transfer_feature,
        position_num_embeddings,
        position_fixed_index,
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

        virtual_position_grad = (
            position_parameter.grad.detach().float().new_zeros(
                position_parameter.numel() + 1
            )
        )
        position_keep = torch.ones(
            virtual_position_grad.numel(),
            dtype=torch.bool,
            device=virtual_position_grad.device,
        )
        position_keep[position_fixed_index] = False
        virtual_position_grad[position_keep] = (
            position_parameter.grad.detach().reshape(-1).float()
        )
        position_fixed_row = position_fixed_index // embedding_dim
        position_row_start = position_fixed_row * embedding_dim
        position_row_end = position_row_start + embedding_dim
        virtual_position_grad[position_fixed_index] = (
            -virtual_position_grad[
                position_row_start:position_row_end
            ].sum()
        )
        total_sq.add_(
            virtual_position_grad[position_fixed_index].pow(2)
        )

        token_matrix = virtual_token_grad.view(
            num_embeddings,
            embedding_dim,
        )
        position_matrix = virtual_position_grad.view(
            position_num_embeddings,
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
        model.pos_emb.fixed_index,
    )
>>>>>>> REPLACE