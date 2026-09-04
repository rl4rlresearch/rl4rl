MECHANISM: Joint token–position common-column gauge

HYPOTHESIS: Fixing feature-4 of the final token row and transferring its offset into every positional row will reduce the model from 1617 to 1616 parameters while retaining at least 99% accuracy, because token-plus-position inputs and output probabilities are unchanged, and the virtual AdamW path reproduces the omitted gradients and updates.

INTENDED_EDIT: Extend the token-embedding quotient by one coordinate, transfer that coordinate into positional embeddings at initialization, and jointly optimize token and positional parameters with both omitted coordinates restored virtually.

EVIDENCE: The current global embedding quotient reaches 99.83%. Unlike the isolated positional-row quotient that fell to 37.73%, this gauge leaves every token-plus-position input exactly unchanged; feature 4 is also the strongest coordinate-specific choice, having supported successful LayerNorm-scale, LayerNorm-bias, and MLP-output gauges.

<<<<<<< SEARCH
class GaugeFixedEmbedding(nn.Module):
    """Embedding with its global all-entries shift fixed to zero."""

    def __init__(self, num_embeddings: int, embedding_dim: int):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim

        # Match nn.Embedding's constructor-time random-number consumption.
        full_weight = torch.empty(num_embeddings, embedding_dim)
        nn.init.normal_(full_weight)
        anchor = full_weight.reshape(-1)[-1].clone()
        fixed = (full_weight - anchor).reshape(-1)[:-1].clone()
        self.weight = nn.Parameter(fixed)

    def full_weight(self) -> torch.Tensor:
        return F.pad(self.weight, (0, 1)).view(
            self.num_embeddings,
            self.embedding_dim,
        )

    @torch.no_grad()
    def reset_from_full_(self, full_weight: torch.Tensor) -> None:
        anchor = full_weight.reshape(-1)[-1].clone()
        self.weight.copy_((full_weight - anchor).reshape(-1)[:-1])

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        return F.embedding(idx, self.full_weight())
=======
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
>>>>>>> REPLACE

<<<<<<< SEARCH
        # The former tied lm_head reinitialized the shared weight last.
        self._init_weights(self.token_emb)
=======
        # The former tied lm_head reinitialized the shared weight last.
        self._init_weights(self.token_emb)
        with torch.no_grad():
            self.pos_emb.weight[:, self.token_emb.transfer_feature].add_(
                self.token_emb.transfer_offset
            )
>>>>>>> REPLACE

<<<<<<< SEARCH
class KeyGaugeAdamW:
=======
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

    def _keep_mask(self) -> torch.Tensor:
        keep = torch.ones(
            self.token_parameter.numel() + 2,
            dtype=torch.bool,
            device=self.token_parameter.device,
        )
        keep[list(self.fixed_indices)] = False
        return keep

    def zero_grad(self, set_to_none: bool = True) -> None:
        for parameter in self.parameters:
            if set_to_none:
                parameter.grad = None
            elif parameter.grad is not None:
                parameter.grad.zero_()

    @torch.no_grad()
    def step(self) -> None:
        if (
            self.token_parameter.grad is None
            or self.position_parameter.grad is None
        ):
            return

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

        state = self.state
        state["step"] += 1
        step = state["step"]

        token_exp_avg = state["token_exp_avg"]
        token_exp_avg_sq = state["token_exp_avg_sq"]
        token_exp_avg.mul_(self.beta1).add_(
            virtual_token_grad,
            alpha=1.0 - self.beta1,
        )
        token_exp_avg_sq.mul_(self.beta2).addcmul_(
            virtual_token_grad,
            virtual_token_grad,
            value=1.0 - self.beta2,
        )

        position_exp_avg = state["position_exp_avg"]
        position_exp_avg_sq = state["position_exp_avg_sq"]
        position_exp_avg.mul_(self.beta1).add_(
            position_grad,
            alpha=1.0 - self.beta1,
        )
        position_exp_avg_sq.mul_(self.beta2).addcmul_(
            position_grad,
            position_grad,
            value=1.0 - self.beta2,
        )

        bias_correction1 = 1.0 - self.beta1**step
        bias_correction2 = 1.0 - self.beta2**step
        token_direction = token_exp_avg / (
            token_exp_avg_sq.sqrt().div(
                math.sqrt(bias_correction2)
            ).add(self.eps)
        )
        position_direction = position_exp_avg / (
            position_exp_avg_sq.sqrt().div(
                math.sqrt(bias_correction2)
            ).add(self.eps)
        )

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


class KeyGaugeAdamW:
>>>>>>> REPLACE

<<<<<<< SEARCH
def clip_grad_norm_with_virtual_gauge(
    model: TinyDecoderLM,
    gauge_parameters,
    key_gauges,
    max_norm: float,
) -> None:
=======
def clip_grad_norm_with_virtual_gauge(
    model: TinyDecoderLM,
    gauge_parameters,
    token_position_gauge,
    key_gauges,
    max_norm: float,
) -> None:
>>>>>>> REPLACE

<<<<<<< SEARCH
    # Recover the omitted key gradients from the LayerNorm-null directions.
=======
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

    # Recover the omitted key gradients from the LayerNorm-null directions.
>>>>>>> REPLACE

<<<<<<< SEARCH
    gauge_parameters = [
        model.token_emb.weight,
        *[block.attn.proj_bias for block in model.blocks],
        *[block.mlp.fc2.bias for block in model.blocks],
    ]
    key_gauges = [
        (
            block.attn.qkv.weight,
            block.ln1.weight,
            model_cfg.d_model,
            block.attn.qkv.fixed_rows,
        )
        for block in model.blocks
    ]
    gauge_parameter_ids = {
        id(p) for p in [
            *gauge_parameters,
            *[parameter for parameter, _, _, _ in key_gauges],
        ]
    }
    ordinary_parameters = [
        p for p in model.parameters() if id(p) not in gauge_parameter_ids
    ]
=======
    token_position_gauge = (
        model.token_emb.weight,
        model.pos_emb.weight,
        model.token_emb.num_embeddings,
        model.token_emb.embedding_dim,
        model.token_emb.transfer_feature,
    )
    gauge_parameters = [
        *[block.attn.proj_bias for block in model.blocks],
        *[block.mlp.fc2.bias for block in model.blocks],
    ]
    key_gauges = [
        (
            block.attn.qkv.weight,
            block.ln1.weight,
            model_cfg.d_model,
            block.attn.qkv.fixed_rows,
        )
        for block in model.blocks
    ]
    gauge_parameter_ids = {
        id(p) for p in [
            token_position_gauge[0],
            token_position_gauge[1],
            *gauge_parameters,
            *[parameter for parameter, _, _, _ in key_gauges],
        ]
    }
    ordinary_parameters = [
        p for p in model.parameters() if id(p) not in gauge_parameter_ids
    ]
>>>>>>> REPLACE

<<<<<<< SEARCH
    gauge_optimizer = GaugeFixedAdamW(
        gauge_parameters,
        lr=train_cfg.lr,
        weight_decay=train_cfg.weight_decay,
    )
    key_optimizer = KeyGaugeAdamW(
=======
    gauge_optimizer = GaugeFixedAdamW(
        gauge_parameters,
        lr=train_cfg.lr,
        weight_decay=train_cfg.weight_decay,
    )
    token_position_optimizer = TokenPositionGaugeAdamW(
        *token_position_gauge,
        lr=train_cfg.lr,
        weight_decay=train_cfg.weight_decay,
    )
    key_optimizer = KeyGaugeAdamW(
>>>>>>> REPLACE

<<<<<<< SEARCH
        gauge_optimizer.lr = lr_now
        key_optimizer.lr = lr_now
=======
        gauge_optimizer.lr = lr_now
        token_position_optimizer.lr = lr_now
        key_optimizer.lr = lr_now
>>>>>>> REPLACE

<<<<<<< SEARCH
        optimizer.zero_grad(set_to_none=True)
        gauge_optimizer.zero_grad(set_to_none=True)
        key_optimizer.zero_grad(set_to_none=True)
=======
        optimizer.zero_grad(set_to_none=True)
        gauge_optimizer.zero_grad(set_to_none=True)
        token_position_optimizer.zero_grad(set_to_none=True)
        key_optimizer.zero_grad(set_to_none=True)
>>>>>>> REPLACE

<<<<<<< SEARCH
            clip_grad_norm_with_virtual_gauge(
                model,
                gauge_parameters,
                key_gauges,
                train_cfg.grad_clip,
            )
=======
            clip_grad_norm_with_virtual_gauge(
                model,
                gauge_parameters,
                token_position_gauge,
                key_gauges,
                train_cfg.grad_clip,
            )
>>>>>>> REPLACE

<<<<<<< SEARCH
        key_optimizer.step()
        optimizer.step()
        gauge_optimizer.step()
=======
        key_optimizer.step()
        optimizer.step()
        gauge_optimizer.step()
        token_position_optimizer.step()
>>>>>>> REPLACE