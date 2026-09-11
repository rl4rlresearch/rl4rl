MECHANISM: One-coordinate coupled token–position translation quotient

HYPOTHESIS: Fixing one additional feature coordinate of the final token row and transferring its column-wide offset into positional embeddings will reduce parameters from 1633 to 1632 while retaining at least 99% accuracy when coupled virtual-coordinate AdamW preserves full token and position gradients, moments, decay, and clipping.

INTENDED_EDIT: Extend the successful tied-embedding scalar gauge by one token–position translation coordinate, reconstruct both omitted token coordinates, transfer the initialization anchor into positional embeddings, and jointly optimize the coupled embeddings in virtual full coordinates.

EVIDENCE: The single tied-embedding gauge achieved 99.72% at 1633 parameters, whereas removing seven additional row-gauge coordinates at once reached only 53.86%; a one-coordinate titration is the smallest informative continuation while preserving the quotient-aware optimization treatment.

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
    """Embedding with global and one token-position shift fixed to zero."""

    def __init__(self, num_embeddings: int, embedding_dim: int):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.gauge_feature = embedding_dim - 2

        # Match nn.Embedding's constructor-time random-number consumption.
        full_weight = torch.empty(num_embeddings, embedding_dim)
        nn.init.normal_(full_weight)
        global_anchor = full_weight[-1, -1].clone()
        fixed = full_weight - global_anchor
        feature_anchor = fixed[-1, self.gauge_feature].clone()
        fixed[:, self.gauge_feature].sub_(feature_anchor)
        self.weight = nn.Parameter(fixed.reshape(-1)[:-2].clone())

    def full_weight(self) -> torch.Tensor:
        return F.pad(self.weight, (0, 2)).view(
            self.num_embeddings,
            self.embedding_dim,
        )

    @torch.no_grad()
    def reset_from_full_(self, full_weight: torch.Tensor) -> torch.Tensor:
        global_anchor = full_weight[-1, -1].clone()
        fixed = full_weight - global_anchor
        feature_anchor = fixed[-1, self.gauge_feature].clone()
        fixed[:, self.gauge_feature].sub_(feature_anchor)
        self.weight.copy_(fixed.reshape(-1)[:-2])
        return feature_anchor

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        return F.embedding(idx, self.full_weight())
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.apply(self._init_weights)
        # The former tied lm_head reinitialized the shared weight last.
        self._init_weights(self.token_emb)
=======
        self.apply(self._init_weights)
        # The former tied lm_head reinitialized the shared weight last.
        full_weight = torch.empty(
            self.token_emb.num_embeddings,
            self.token_emb.embedding_dim,
            device=self.token_emb.weight.device,
            dtype=self.token_emb.weight.dtype,
        )
        nn.init.normal_(full_weight, mean=0.0, std=0.02)
        feature_anchor = self.token_emb.reset_from_full_(full_weight)
        with torch.no_grad():
            self.pos_emb.weight[:, self.token_emb.gauge_feature].add_(
                feature_anchor
            )
>>>>>>> REPLACE

<<<<<<< SEARCH
            p.add_(
                quotient_direction.view_as(p),
                alpha=-self.lr / bias_correction1,
            )


def clip_grad_norm_with_virtual_gauge(
    model: TinyDecoderLM,
    gauge_parameters,
    max_norm: float,
) -> None:
=======
            p.add_(
                quotient_direction.view_as(p),
                alpha=-self.lr / bias_correction1,
            )


class CoupledEmbeddingAdamW:
    """AdamW on full token/position coordinates modulo two exact shifts."""

    def __init__(
        self,
        token_parameter,
        position_parameter,
        num_embeddings: int,
        embedding_dim: int,
        lr: float,
        weight_decay: float,
        betas=(0.9, 0.999),
        eps: float = 1e-8,
    ):
        self.token_parameter = token_parameter
        self.position_parameter = position_parameter
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.gauge_feature = embedding_dim - 2
        self.lr = lr
        self.weight_decay = weight_decay
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.token_state = {
            "step": 0,
            "exp_avg": torch.zeros(
                num_embeddings,
                embedding_dim,
                device=token_parameter.device,
                dtype=token_parameter.dtype,
            ),
            "exp_avg_sq": torch.zeros(
                num_embeddings,
                embedding_dim,
                device=token_parameter.device,
                dtype=token_parameter.dtype,
            ),
        }
        self.position_state = {
            "step": 0,
            "exp_avg": torch.zeros_like(position_parameter),
            "exp_avg_sq": torch.zeros_like(position_parameter),
        }

    def zero_grad(self, set_to_none: bool = True) -> None:
        for p in (self.token_parameter, self.position_parameter):
            if set_to_none:
                p.grad = None
            elif p.grad is not None:
                p.grad.zero_()

    def _virtual_gradients(self):
        if (
            self.token_parameter.grad is None
            or self.position_parameter.grad is None
        ):
            return None, None

        stored_grad = self.token_parameter.grad.detach().reshape(-1)
        token_grad = torch.cat(
            (stored_grad, stored_grad.new_zeros(2))
        ).view(self.num_embeddings, self.embedding_dim)
        position_grad = self.position_parameter.grad.detach()

        # Token-column and position-column shifts leave inputs unchanged and
        # alter output logits only by a vocabulary-wide constant.
        token_grad[-1, self.gauge_feature] = (
            position_grad[:, self.gauge_feature].sum()
            - token_grad[:, self.gauge_feature].sum()
        )
        # A global shift of every token-embedding entry is also invariant.
        token_grad[-1, -1] = -token_grad.sum()
        return token_grad, position_grad

    def omitted_gradient_sq(self) -> torch.Tensor:
        token_grad, _ = self._virtual_gradients()
        if token_grad is None:
            return torch.zeros(
                (),
                device=self.token_parameter.device,
                dtype=torch.float32,
            )
        return token_grad[-1, -2:].float().pow(2).sum()

    def _direction(self, state, grad):
        state["step"] += 1
        step = state["step"]
        exp_avg = state["exp_avg"]
        exp_avg_sq = state["exp_avg_sq"]
        exp_avg.mul_(self.beta1).add_(grad, alpha=1.0 - self.beta1)
        exp_avg_sq.mul_(self.beta2).addcmul_(
            grad, grad, value=1.0 - self.beta2
        )
        bias_correction1 = 1.0 - self.beta1**step
        bias_correction2 = 1.0 - self.beta2**step
        denom = exp_avg_sq.sqrt().div_(
            math.sqrt(bias_correction2)
        ).add_(self.eps)
        return exp_avg / denom, bias_correction1

    @torch.no_grad()
    def step(self) -> None:
        token_grad, position_grad = self._virtual_gradients()
        if token_grad is None:
            return

        token_direction, token_correction = self._direction(
            self.token_state, token_grad
        )
        position_direction, position_correction = self._direction(
            self.position_state, position_grad
        )

        token_full = torch.cat(
            (
                self.token_parameter,
                self.token_parameter.new_zeros(2),
            )
        ).view(self.num_embeddings, self.embedding_dim)
        decay = 1.0 - self.lr * self.weight_decay
        token_new = token_full * decay
        token_new.add_(
            token_direction,
            alpha=-self.lr / token_correction,
        )
        position_new = self.position_parameter * decay
        position_new.add_(
            position_direction,
            alpha=-self.lr / position_correction,
        )

        global_anchor = token_new[-1, -1].clone()
        token_new.sub_(global_anchor)
        feature_anchor = token_new[-1, self.gauge_feature].clone()
        token_new[:, self.gauge_feature].sub_(feature_anchor)
        position_new[:, self.gauge_feature].add_(feature_anchor)

        self.token_parameter.copy_(token_new.reshape(-1)[:-2])
        self.position_parameter.copy_(position_new)


def clip_grad_norm_with_virtual_gauge(
    model: TinyDecoderLM,
    embedding_optimizer: CoupledEmbeddingAdamW,
    gauge_parameters,
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
    # Add gradients of coordinates omitted by the exact gauge choices.
    total_sq.add_(embedding_optimizer.omitted_gradient_sq())
    for p in gauge_parameters:
        if p.grad is not None:
            total_sq.add_(p.grad.detach().float().sum().pow(2))

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
    optimizer = torch.optim.AdamW(
        ordinary_parameters,
        lr=train_cfg.lr,
        weight_decay=train_cfg.weight_decay,
    )
    gauge_optimizer = GaugeFixedAdamW(
        gauge_parameters,
        lr=train_cfg.lr,
        weight_decay=train_cfg.weight_decay,
    )
=======
    embedding_parameters = [
        model.token_emb.weight,
        model.pos_emb.weight,
    ]
    gauge_parameters = [
        block.attn.proj_bias for block in model.blocks
    ]
    special_parameter_ids = {
        id(p) for p in [*embedding_parameters, *gauge_parameters]
    }
    ordinary_parameters = [
        p for p in model.parameters() if id(p) not in special_parameter_ids
    ]
    optimizer = torch.optim.AdamW(
        ordinary_parameters,
        lr=train_cfg.lr,
        weight_decay=train_cfg.weight_decay,
    )
    embedding_optimizer = CoupledEmbeddingAdamW(
        model.token_emb.weight,
        model.pos_emb.weight,
        model.token_emb.num_embeddings,
        model.token_emb.embedding_dim,
        lr=train_cfg.lr,
        weight_decay=train_cfg.weight_decay,
    )
    gauge_optimizer = GaugeFixedAdamW(
        gauge_parameters,
        lr=train_cfg.lr,
        weight_decay=train_cfg.weight_decay,
    )
>>>>>>> REPLACE

<<<<<<< SEARCH
        for pg in optimizer.param_groups:
            pg["lr"] = lr_now
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
        for pg in optimizer.param_groups:
            pg["lr"] = lr_now
        embedding_optimizer.lr = lr_now
        gauge_optimizer.lr = lr_now

        _, loss = model(x, y)
        optimizer.zero_grad(set_to_none=True)
        embedding_optimizer.zero_grad(set_to_none=True)
        gauge_optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if train_cfg.grad_clip > 0:
            clip_grad_norm_with_virtual_gauge(
                model,
                embedding_optimizer,
                gauge_parameters,
                train_cfg.grad_clip,
            )
        optimizer.step()
        embedding_optimizer.step()
        gauge_optimizer.step()
>>>>>>> REPLACE