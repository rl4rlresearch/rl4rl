MECHANISM: Token–position additive gauge with a virtual AdamW coordinate

HYPOTHESIS: Fixing positional coordinate `[0, 0]` and folding its virtual optimizer update through the token/position additive symmetry will reduce the model from 1622 to 1621 parameters while maintaining at least 99% accuracy.

INTENDED_EDIT: Replace the positional embedding with a one-coordinate gauge-fixed embedding, preserve the original functional initialization by transferring its anchor into the tied token embedding, and reconstruct the omitted coordinate’s gradient and AdamW update during training.

EVIDENCE: The 1622-parameter design achieved 99.92%; the 1599-parameter positional-row experiment removed 23 coordinates simultaneously and reached 73.41%, motivating a single exact positional gauge with retained virtual optimization geometry.

<<<<<<< SEARCH
    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        return F.embedding(idx, self.full_weight())


class GaugeFixedQKV(nn.Module):
=======
    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        return F.embedding(idx, self.full_weight())


class GaugeFixedPositionalEmbedding(nn.Module):
    """Position embedding with one token-position additive gauge removed."""

    def __init__(self, num_embeddings: int, embedding_dim: int):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim

        # Match nn.Embedding's constructor-time random-number consumption.
        source = nn.Embedding(num_embeddings, embedding_dim)
        full_weight = source.weight.detach().clone()
        reduced, anchor = self._reduce(full_weight)
        self.weight = nn.Parameter(reduced)
        self.initial_anchor = anchor

    def _reduce(
        self,
        full_weight: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        anchor = full_weight[0, 0].clone()
        gauged = full_weight.clone()
        gauged[:, 0].sub_(anchor)
        return gauged.reshape(-1)[1:].clone(), anchor

    def full_weight(self) -> torch.Tensor:
        return F.pad(self.weight, (1, 0)).view(
            self.num_embeddings,
            self.embedding_dim,
        )

    @torch.no_grad()
    def reset_from_full_(self, full_weight: torch.Tensor) -> None:
        reduced, anchor = self._reduce(full_weight)
        self.weight.copy_(reduced)
        self.initial_anchor = anchor

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        return F.embedding(idx, self.full_weight())


class GaugeFixedQKV(nn.Module):
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.pos_emb = nn.Embedding(cfg.max_seq_len, cfg.d_model)
=======
        self.pos_emb = GaugeFixedPositionalEmbedding(
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
        elif isinstance(module, GaugeFixedPositionalEmbedding):
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
        self.apply(self._init_weights)
        # The former tied lm_head reinitialized the shared weight last.
        self._init_weights(self.token_emb)
=======
        self.apply(self._init_weights)
        # The former tied lm_head reinitialized the shared weight last.
        self._init_weights(self.token_emb)

        # Transfer the fixed positional coordinate into every token row.
        # This preserves input sums, while its tied-output effect is only a
        # common shift of all vocabulary logits.
        full_token_weight = self.token_emb.full_weight().detach().clone()
        full_token_weight[:, 0].add_(self.pos_emb.initial_anchor)
        self.token_emb.reset_from_full_(full_token_weight)
        self.pos_emb.initial_anchor = None
>>>>>>> REPLACE

<<<<<<< SEARCH
            p.add_(
                quotient_direction.view_as(p),
                alpha=-self.lr / bias_correction1,
            )


class KeyGaugeAdamW:
=======
            p.add_(
                quotient_direction.view_as(p),
                alpha=-self.lr / bias_correction1,
            )


class TokenPositionGaugeAdamW:
    """AdamW retaining the positional coordinate absorbed by token embeddings."""

    def __init__(
        self,
        gauges,
        lr: float,
        weight_decay: float,
        betas=(0.9, 0.999),
        eps: float = 1e-8,
    ):
        self.gauges = list(gauges)
        self.parameters = [
            position_parameter
            for position_parameter, _, _ in self.gauges
        ]
        self.lr = lr
        self.weight_decay = weight_decay
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.state = {
            position_parameter: {
                "step": 0,
                "exp_avg": torch.zeros(
                    position_parameter.numel() + 1,
                    device=position_parameter.device,
                    dtype=position_parameter.dtype,
                ),
                "exp_avg_sq": torch.zeros(
                    position_parameter.numel() + 1,
                    device=position_parameter.device,
                    dtype=position_parameter.dtype,
                ),
            }
            for position_parameter, _, _ in self.gauges
        }

    @staticmethod
    def omitted_gradient(
        position_parameter,
        token_parameter,
        d_model: int,
    ) -> torch.Tensor:
        position_grad = position_parameter.grad.detach().reshape(-1)
        token_grad = token_parameter.grad.detach().reshape(-1)
        token_feature_sum = token_grad[0::d_model].sum()
        other_position_sum = position_grad[d_model - 1 :: d_model].sum()
        return token_feature_sum - other_position_sum

    def zero_grad(self, set_to_none: bool = True) -> None:
        for parameter in self.parameters:
            if set_to_none:
                parameter.grad = None
            elif parameter.grad is not None:
                parameter.grad.zero_()

    @torch.no_grad()
    def step(self) -> None:
        for position_parameter, token_parameter, d_model in self.gauges:
            if (
                position_parameter.grad is None
                or token_parameter.grad is None
            ):
                continue

            grad = position_parameter.grad.detach().reshape(-1)
            virtual_grad = torch.cat(
                (
                    self.omitted_gradient(
                        position_parameter,
                        token_parameter,
                        d_model,
                    ).reshape(1),
                    grad,
                )
            )

            state = self.state[position_parameter]
            state["step"] += 1
            step = state["step"]
            exp_avg = state["exp_avg"]
            exp_avg_sq = state["exp_avg_sq"]
            exp_avg.mul_(self.beta1).add_(
                virtual_grad,
                alpha=1.0 - self.beta1,
            )
            exp_avg_sq.mul_(self.beta2).addcmul_(
                virtual_grad,
                virtual_grad,
                value=1.0 - self.beta2,
            )

            bias_correction1 = 1.0 - self.beta1**step
            bias_correction2 = 1.0 - self.beta2**step
            denom = exp_avg_sq.sqrt().div_(
                math.sqrt(bias_correction2)
            ).add_(self.eps)
            direction = exp_avg / denom

            quotient_direction = direction[1:].clone()
            quotient_direction[d_model - 1 :: d_model].sub_(direction[0])

            position_parameter.mul_(1.0 - self.lr * self.weight_decay)
            position_parameter.add_(
                quotient_direction.view_as(position_parameter),
                alpha=-self.lr / bias_correction1,
            )

            token_parameter.view(-1)[0::d_model].add_(
                direction[0],
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
    position_gauges,
    key_gauges,
    max_norm: float,
) -> None:
>>>>>>> REPLACE

<<<<<<< SEARCH
    # Recover the omitted key gradients from the LayerNorm-null directions.
    for parameter, ln_scale, d_model, fixed_rows in key_gauges:
=======
    # Recover the positional gradient absorbed into the token embedding.
    for position_parameter, token_parameter, d_model in position_gauges:
        if (
            position_parameter.grad is not None
            and token_parameter.grad is not None
        ):
            omitted = TokenPositionGaugeAdamW.omitted_gradient(
                position_parameter,
                token_parameter,
                d_model,
            )
            total_sq.add_(omitted.detach().float().pow(2))

    # Recover the omitted key gradients from the LayerNorm-null directions.
    for parameter, ln_scale, d_model, fixed_rows in key_gauges:
>>>>>>> REPLACE

<<<<<<< SEARCH
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
=======
    key_gauges = [
        (
            block.attn.qkv.weight,
            block.ln1.weight,
            model_cfg.d_model,
            block.attn.qkv.fixed_rows,
        )
        for block in model.blocks
    ]
    position_gauges = [
        (
            model.pos_emb.weight,
            model.token_emb.weight,
            model_cfg.d_model,
        )
    ]
    gauge_parameter_ids = {
        id(p) for p in [
            *gauge_parameters,
            *[
                position_parameter
                for position_parameter, _, _ in position_gauges
            ],
            *[parameter for parameter, _, _, _ in key_gauges],
        ]
    }
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
    position_optimizer = TokenPositionGaugeAdamW(
        position_gauges,
        lr=train_cfg.lr,
        weight_decay=train_cfg.weight_decay,
    )
    key_optimizer = KeyGaugeAdamW(
>>>>>>> REPLACE

<<<<<<< SEARCH
        gauge_optimizer.lr = lr_now
        key_optimizer.lr = lr_now

        _, loss = model(x, y)
        optimizer.zero_grad(set_to_none=True)
        gauge_optimizer.zero_grad(set_to_none=True)
        key_optimizer.zero_grad(set_to_none=True)
=======
        gauge_optimizer.lr = lr_now
        position_optimizer.lr = lr_now
        key_optimizer.lr = lr_now

        _, loss = model(x, y)
        optimizer.zero_grad(set_to_none=True)
        gauge_optimizer.zero_grad(set_to_none=True)
        position_optimizer.zero_grad(set_to_none=True)
        key_optimizer.zero_grad(set_to_none=True)
>>>>>>> REPLACE

<<<<<<< SEARCH
            clip_grad_norm_with_virtual_gauge(
                model,
                gauge_parameters,
                key_gauges,
                train_cfg.grad_clip,
            )
        key_optimizer.step()
        optimizer.step()
        gauge_optimizer.step()
=======
            clip_grad_norm_with_virtual_gauge(
                model,
                gauge_parameters,
                position_gauges,
                key_gauges,
                train_cfg.grad_clip,
            )
        key_optimizer.step()
        optimizer.step()
        gauge_optimizer.step()
        position_optimizer.step()
>>>>>>> REPLACE