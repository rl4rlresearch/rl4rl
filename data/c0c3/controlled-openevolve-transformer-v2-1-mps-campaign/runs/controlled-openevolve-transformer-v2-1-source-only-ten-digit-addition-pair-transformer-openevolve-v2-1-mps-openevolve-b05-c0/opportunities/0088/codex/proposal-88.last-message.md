MECHANISM: Virtual-AdamW quotient of a per-position residual-stream shift

HYPOTHESIS: Gauge-fixing one positional-embedding coordinate will reduce the verified model from 1607 to 1606 parameters while retaining at least 99% accuracy, because adding a common feature shift at one position is exactly invisible to the pre-norm block and final LayerNorm, while reconstructed gradients and full-coordinate AdamW moments preserve optimizer dynamics.

INTENDED_EDIT: Replace the positional embedding with a one-coordinate gauge-fixed embedding and extend the coupled token-position optimizer and gradient clipping to virtually optimize and project the omitted positional coordinate.

EVIDENCE: The 1607-parameter design achieved 99.97% using several virtual-AdamW quotients, while attempts to extend the key-row, token-transfer, and LayerNorm-scale reductions failed. This tests a distinct exact symmetry with the optimizer-state preservation that previously rescued otherwise destructive coordinate removal.

<<<<<<< SEARCH
    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        return F.embedding(idx, self.full_weight())


class GaugeFixedQKV(nn.Module):
=======
    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        return F.embedding(idx, self.full_weight())


class GaugeFixedPositionEmbedding(nn.Module):
    """Position embedding with one residual-stream shift gauge fixed."""

    def __init__(self, num_embeddings: int, embedding_dim: int):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.fixed_row = num_embeddings - 1
        self.fixed_feature = embedding_dim - 1
        self.fixed_index = (
            self.fixed_row * embedding_dim + self.fixed_feature
        )

        # Match nn.Embedding's constructor-time random-number consumption.
        full_weight = torch.empty(num_embeddings, embedding_dim)
        nn.init.normal_(full_weight)
        self.weight = nn.Parameter(self._reduce(full_weight))

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
        elif isinstance(module, GaugeFixedQKV):
            full_weight = torch.empty(
=======
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
            full_weight = torch.empty(
>>>>>>> REPLACE

<<<<<<< SEARCH
        with torch.no_grad():
            self.pos_emb.weight[:, self.token_emb.transfer_feature].add_(
                self.token_emb.transfer_offset
            )
=======
        with torch.no_grad():
            full_position_weight = self.pos_emb.full_weight()
            full_position_weight[
                :,
                self.token_emb.transfer_feature,
            ].add_(self.token_emb.transfer_offset)
            self.pos_emb.reset_from_full_(full_position_weight)
>>>>>>> REPLACE

<<<<<<< SEARCH
class TokenPositionGaugeAdamW:
    """AdamW with virtual coordinates for two coupled embedding gauges."""
=======
class TokenPositionGaugeAdamW:
    """AdamW for coupled token gauges and a positional shift gauge."""
>>>>>>> REPLACE

<<<<<<< SEARCH
        embedding_dim: int,
        transfer_feature: int,
        lr: float,
=======
        embedding_dim: int,
        transfer_feature: int,
        num_positions: int,
        position_fixed_index: int,
        lr: float,
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.embedding_dim = embedding_dim
        self.transfer_feature = transfer_feature
        self.transfer_index = (
=======
        self.embedding_dim = embedding_dim
        self.transfer_feature = transfer_feature
        self.num_positions = num_positions
        self.position_fixed_index = position_fixed_index
        self.transfer_index = (
>>>>>>> REPLACE

<<<<<<< SEARCH
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
    def zero_grad(self, set_to_none: bool = True) -> None:
=======
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
        position_fixed_row = (
            self.position_fixed_index // self.embedding_dim
        )
        position_matrix = virtual_position_grad.view(
            self.num_positions,
            self.embedding_dim,
        )
        virtual_position_grad[self.position_fixed_index] = (
            -position_matrix[position_fixed_row].sum()
        )

        token_matrix = virtual_token_grad.view(
            self.num_embeddings,
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
            self.num_positions,
            self.embedding_dim,
        ).clone()
        quotient_position[:, self.transfer_feature].add_(
            transfer_direction - global_direction
        )
        position_anchor_direction = quotient_position.reshape(-1)[
            self.position_fixed_index
        ].clone()
        quotient_position[position_fixed_row].sub_(
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
        transfer_feature,
    ) = token_position_gauge
=======
        transfer_feature,
        num_positions,
        position_fixed_index,
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

        position_keep = torch.ones(
            position_parameter.numel() + 1,
            dtype=torch.bool,
            device=position_parameter.device,
        )
        position_keep[position_fixed_index] = False
        virtual_position_grad = (
            position_parameter.grad.detach().float().new_zeros(
                position_keep.numel()
            )
        )
        virtual_position_grad[position_keep] = (
            position_parameter.grad.detach().reshape(-1).float()
        )
        position_fixed_row = position_fixed_index // embedding_dim
        position_matrix = virtual_position_grad.view(
            num_positions,
            embedding_dim,
        )
        virtual_position_grad[position_fixed_index] = (
            -position_matrix[position_fixed_row].sum()
        )
        total_sq.add_(
            virtual_position_grad[position_fixed_index].pow(2)
        )

        token_matrix = virtual_token_grad.view(
            num_embeddings,
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
        model.token_emb.embedding_dim,
        model.token_emb.transfer_feature,
    )
=======
        model.token_emb.embedding_dim,
        model.token_emb.transfer_feature,
        model.pos_emb.num_embeddings,
        model.pos_emb.fixed_index,
    )
>>>>>>> REPLACE