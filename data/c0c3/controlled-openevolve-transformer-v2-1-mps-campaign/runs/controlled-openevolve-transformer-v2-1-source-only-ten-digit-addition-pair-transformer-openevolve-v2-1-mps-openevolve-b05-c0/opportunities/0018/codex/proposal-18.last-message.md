MECHANISM: Coupled tied-token/position row-gauge quotient

HYPOTHESIS: Fixing the final token-embedding row to zero will reduce the model from 1633 to 1626 parameters while retaining at least 99% accuracy, because any removed eight-dimensional row can be transferred to every positional vector without changing transformer inputs, while it changes output logits only by a vocabulary-wide constant; coupled virtual AdamW preserves the full-coordinate optimization dynamics.

INTENDED_EDIT: Replace the single-scalar tied-embedding gauge with an eight-coordinate row gauge, transfer its initialization anchor into the positional embedding, and jointly optimize token and positional embeddings using reconstructed full gradients, moments, weight decay, and clipping.

EVIDENCE: The quotient-aware attention-bias model achieved 99.85% at 1634 parameters and the tied-embedding scalar quotient achieved 99.72% at 1633, showing virtual full-coordinate AdamW can preserve accuracy across exact gauges. The independent positional gauge failed at 68.31%, motivating this coupled gauge, which leaves the summed token-plus-position input unchanged rather than quotienting position embeddings alone.

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
    """Tied embedding with its final vocabulary row fixed to zero."""

    def __init__(self, num_embeddings: int, embedding_dim: int):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim

        # Match nn.Embedding's constructor-time random-number consumption.
        full_weight = torch.empty(num_embeddings, embedding_dim)
        nn.init.normal_(full_weight)
        anchor = full_weight[-1].clone()
        self.weight = nn.Parameter((full_weight[:-1] - anchor).clone())
        self._last_anchor = anchor

    def full_weight(self) -> torch.Tensor:
        return F.pad(self.weight, (0, 0, 0, 1))

    @torch.no_grad()
    def reset_from_full_(self, full_weight: torch.Tensor) -> None:
        anchor = full_weight[-1].clone()
        self.weight.copy_(full_weight[:-1] - anchor)
        self._last_anchor = anchor

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        return F.embedding(idx, self.full_weight())
>>>>>>> REPLACE

<<<<<<< SEARCH
        # The former tied lm_head reinitialized the shared weight last.
        self._init_weights(self.token_emb)
=======
        # The former tied lm_head reinitialized the shared weight last.
        self._init_weights(self.token_emb)
        # Transfer the removed row to position vectors so transformer inputs
        # exactly represent the corresponding full-embedding initialization.
        with torch.no_grad():
            self.pos_emb.weight.add_(self.token_emb._last_anchor)
        del self.token_emb._last_anchor
>>>>>>> REPLACE

<<<<<<< SEARCH
class GaugeFixedAdamW:
    """AdamW on a bias quotient while retaining virtual full-bias moments."""
=======
class TiedEmbeddingGaugeAdamW:
    """Coupled AdamW for a zero-row tied embedding and position embedding."""

    def __init__(
        self,
        token_weight,
        position_weight,
        lr: float,
        weight_decay: float,
        betas=(0.9, 0.999),
        eps: float = 1e-8,
    ):
        self.token_weight = token_weight
        self.position_weight = position_weight
        self.lr = lr
        self.weight_decay = weight_decay
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.step_count = 0
        token_shape = (token_weight.size(0) + 1, token_weight.size(1))
        self.token_exp_avg = torch.zeros(
            token_shape, device=token_weight.device, dtype=token_weight.dtype
        )
        self.token_exp_avg_sq = torch.zeros_like(self.token_exp_avg)
        self.position_exp_avg = torch.zeros_like(position_weight)
        self.position_exp_avg_sq = torch.zeros_like(position_weight)

    def zero_grad(self, set_to_none: bool = True) -> None:
        for p in (self.token_weight, self.position_weight):
            if set_to_none:
                p.grad = None
            elif p.grad is not None:
                p.grad.zero_()

    @torch.no_grad()
    def step(self) -> None:
        token_grad = self.token_weight.grad
        position_grad = self.position_weight.grad
        if token_grad is None or position_grad is None:
            return

        # Gauge invariance determines the omitted token-row gradient.
        missing_grad = position_grad.detach().sum(dim=0) - token_grad.detach().sum(dim=0)
        virtual_token_grad = torch.cat(
            (token_grad.detach(), missing_grad.unsqueeze(0)),
            dim=0,
        )
        position_grad = position_grad.detach()

        self.step_count += 1
        self.token_exp_avg.mul_(self.beta1).add_(
            virtual_token_grad, alpha=1.0 - self.beta1
        )
        self.token_exp_avg_sq.mul_(self.beta2).addcmul_(
            virtual_token_grad,
            virtual_token_grad,
            value=1.0 - self.beta2,
        )
        self.position_exp_avg.mul_(self.beta1).add_(
            position_grad, alpha=1.0 - self.beta1
        )
        self.position_exp_avg_sq.mul_(self.beta2).addcmul_(
            position_grad,
            position_grad,
            value=1.0 - self.beta2,
        )

        bias_correction1 = 1.0 - self.beta1**self.step_count
        bias_correction2 = 1.0 - self.beta2**self.step_count
        token_denom = self.token_exp_avg_sq.sqrt().div_(
            math.sqrt(bias_correction2)
        ).add_(self.eps)
        position_denom = self.position_exp_avg_sq.sqrt().div_(
            math.sqrt(bias_correction2)
        ).add_(self.eps)
        token_direction = self.token_exp_avg / token_denom
        position_direction = self.position_exp_avg / position_denom

        decay = 1.0 - self.lr * self.weight_decay
        self.token_weight.mul_(decay)
        self.position_weight.mul_(decay)

        omitted_direction = token_direction[-1]
        self.token_weight.add_(
            token_direction[:-1] - omitted_direction,
            alpha=-self.lr / bias_correction1,
        )
        self.position_weight.add_(
            position_direction + omitted_direction,
            alpha=-self.lr / bias_correction1,
        )


class GaugeFixedAdamW:
    """AdamW on a bias quotient while retaining virtual full-bias moments."""
>>>>>>> REPLACE

<<<<<<< SEARCH
def clip_grad_norm_with_virtual_gauge(
    model: TinyDecoderLM,
    gauge_parameters,
    max_norm: float,
) -> None:
=======
def clip_grad_norm_with_virtual_gauge(
    model: TinyDecoderLM,
    gauge_parameters,
    token_weight,
    position_weight,
    max_norm: float,
) -> None:
>>>>>>> REPLACE

<<<<<<< SEARCH
    # The omitted full-bias gradient is fixed by common-shift invariance.
    for p in gauge_parameters:
        if p.grad is not None:
            total_sq.add_(p.grad.detach().float().sum().pow(2))

    coefficient = min(
=======
    # Each omitted full-bias gradient is fixed by common-shift invariance.
    for p in gauge_parameters:
        if p.grad is not None:
            total_sq.add_(p.grad.detach().float().sum().pow(2))

    # Reconstruct the omitted token row from the coupled token/position gauge.
    if token_weight.grad is not None and position_weight.grad is not None:
        missing_grad = (
            position_weight.grad.detach().float().sum(dim=0)
            - token_weight.grad.detach().float().sum(dim=0)
        )
        total_sq.add_(missing_grad.pow(2).sum())

    coefficient = min(
>>>>>>> REPLACE

<<<<<<< SEARCH
    gauge_parameters = [
        model.token_emb.weight,
        *[block.attn.proj_bias for block in model.blocks],
    ]
    gauge_parameter_ids = {id(p) for p in gauge_parameters}
    ordinary_parameters = [
        p for p in model.parameters() if id(p) not in gauge_parameter_ids
    ]
=======
    gauge_parameters = [
        block.attn.proj_bias for block in model.blocks
    ]
    embedding_gauge_parameters = [
        model.token_emb.weight,
        model.pos_emb.weight,
    ]
    gauge_parameter_ids = {
        id(p) for p in [*gauge_parameters, *embedding_gauge_parameters]
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

    sampler = TrainBatchSampler(train_cfg.batch_size, train_cfg.seed + 1337, reserved_hashes)
=======
    gauge_optimizer = GaugeFixedAdamW(
        gauge_parameters,
        lr=train_cfg.lr,
        weight_decay=train_cfg.weight_decay,
    )
    embedding_optimizer = TiedEmbeddingGaugeAdamW(
        model.token_emb.weight,
        model.pos_emb.weight,
        lr=train_cfg.lr,
        weight_decay=train_cfg.weight_decay,
    )

    sampler = TrainBatchSampler(train_cfg.batch_size, train_cfg.seed + 1337, reserved_hashes)
>>>>>>> REPLACE

<<<<<<< SEARCH
        gauge_optimizer.lr = lr_now

        _, loss = model(x, y)
        optimizer.zero_grad(set_to_none=True)
        gauge_optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if train_cfg.grad_clip > 0:
            clip_grad_norm_with_virtual_gauge(
                model,
                gauge_parameters,
                train_cfg.grad_clip,
            )
        optimizer.step()
        gauge_optimizer.step()
=======
        gauge_optimizer.lr = lr_now
        embedding_optimizer.lr = lr_now

        _, loss = model(x, y)
        optimizer.zero_grad(set_to_none=True)
        gauge_optimizer.zero_grad(set_to_none=True)
        embedding_optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if train_cfg.grad_clip > 0:
            clip_grad_norm_with_virtual_gauge(
                model,
                gauge_parameters,
                model.token_emb.weight,
                model.pos_emb.weight,
                train_cfg.grad_clip,
            )
        optimizer.step()
        gauge_optimizer.step()
        embedding_optimizer.step()
>>>>>>> REPLACE