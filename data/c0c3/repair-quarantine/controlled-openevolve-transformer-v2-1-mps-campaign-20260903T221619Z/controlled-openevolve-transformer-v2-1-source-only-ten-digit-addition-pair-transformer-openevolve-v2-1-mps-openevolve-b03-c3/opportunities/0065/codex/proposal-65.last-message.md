MECHANISM: Seven-dimensional joint token-position translation quotient with ambient AdamW

HYPOTHESIS: Gauge-fixing all seven feature translations independent of the existing scalar gauges will produce a 1,523-parameter model with at least 99% accuracy because it leaves every transformer input and every logit difference unchanged while preserving the qualified 1,530-parameter optimizer trajectory in virtual ambient token and position tables.

INTENDED_EDIT: Fix the first positional vector to zero, absorb its seven effective coordinates into every token vector, and train virtual pre-quotient token and positional tables with full AdamW moments, clipping, gauge projection, and per-step materialization.

EVIDENCE: The current 1,530-parameter design achieved 99.93%, and its tied-token and positional ambient gauges already demonstrate successful embedding-side quotient optimization. The prior joint-translation attempt was unverifiable rather than an observed accuracy failure, motivating a complete implementation of the same exact symmetry.

<<<<<<< SEARCH
        self.weight.copy_(flat[:-1] - flat[-1])

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
=======
        self.weight.copy_(flat[:-1] - flat[-1])

    @torch.no_grad()
    def translate_features_(self, translation: torch.Tensor) -> None:
        full_weight = torch.cat(
            (self.weight, self.weight.new_zeros(1))
        ).view(self.num_embeddings, self.embedding_dim)
        full_weight.add_(translation.unsqueeze(0))
        self.weight.copy_(full_weight.reshape(-1)[:-1])

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
>>>>>>> REPLACE

<<<<<<< SEARCH
class GaugeFixedPositionEmbedding(nn.Module):
    """Position embedding with one shift-invariant scalar removed."""

    def __init__(self, num_embeddings: int, embedding_dim: int):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.first = nn.Parameter(torch.empty(embedding_dim - 1))
        self.rest = nn.Parameter(torch.empty(num_embeddings - 1, embedding_dim))
        self.full_first = None
        self.reset_parameters()

    @torch.no_grad()
    def reset_parameters(self, std: float = 1.0) -> None:
        raw = self.first.new_empty(self.num_embeddings, self.embedding_dim)
        nn.init.normal_(raw, mean=0.0, std=std)
        self.first.copy_(raw[0, :-1] - raw[0, -1])
        self.rest.copy_(raw[1:])

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        first = torch.cat((self.first, self.first.new_zeros(1)))
        if torch.is_grad_enabled():
            first.retain_grad()
            self.full_first = first
        weight = torch.cat((first.unsqueeze(0), self.rest), dim=0)
        return F.embedding(idx, weight)
=======
class GaugeFixedPositionEmbedding(nn.Module):
    """Position embedding with its first vector fixed by joint translation."""

    def __init__(self, num_embeddings: int, embedding_dim: int):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.rest = nn.Parameter(torch.empty(num_embeddings - 1, embedding_dim))
        self.register_buffer(
            "initial_translation",
            torch.zeros(embedding_dim),
            persistent=False,
        )
        self.full_weight = None
        self.reset_parameters()

    @torch.no_grad()
    def reset_parameters(self, std: float = 1.0) -> None:
        raw = self.rest.new_empty(self.num_embeddings, self.embedding_dim)
        nn.init.normal_(raw, mean=0.0, std=std)
        translation = raw[0] - raw[0, -1]
        self.initial_translation.copy_(translation)
        self.rest.copy_(raw[1:] - translation.unsqueeze(0))

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        first = self.rest.new_zeros(1, self.embedding_dim)
        weight = torch.cat((first, self.rest), dim=0)
        if torch.is_grad_enabled():
            weight.retain_grad()
            self.full_weight = weight
        return F.embedding(idx, weight)
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.apply(self._init_weights)

    @staticmethod
=======
        self.apply(self._init_weights)
        self.token_emb.translate_features_(
            self.pos_emb.initial_translation
        )

    @staticmethod
>>>>>>> REPLACE

<<<<<<< SEARCH
    # Preserve full ambient-coordinate AdamW dynamics for the tied-token,
    # positional, three attention-weight, attention-bias, terminal-bias, and six
    # terminal-weight gauges.
    gauge_params = [model.token_emb.weight, model.pos_emb.first]
    for blk in model.blocks:
        gauge_params.extend(blk.attn.proj.weight_prefix)
        gauge_params.append(blk.attn.proj.bias)
        gauge_params.append(blk.mlp.fc2.bias)
        gauge_params.extend(blk.mlp.fc2.weight_prefix)
    gauge_ids = {id(p) for p in gauge_params}
    optimizer = torch.optim.AdamW(
        (p for p in model.parameters() if id(p) not in gauge_ids),
        lr=train_cfg.lr,
        weight_decay=train_cfg.weight_decay,
    )
    gauge_m = [
        torch.zeros(p.numel() + 1, device=device, dtype=p.dtype)
        for p in gauge_params
    ]
    gauge_v = [torch.zeros_like(moment) for moment in gauge_m]
    gauge_step = 0
=======
    # The model stores token/position tables after quotienting their seven
    # joint feature translations. Reconstruct the qualified pre-quotient
    # representative and retain its full AdamW state during training.
    with torch.no_grad():
        initial_translation = (
            model.pos_emb.initial_translation.detach()
        )
        effective_token = torch.cat(
            (
                model.token_emb.weight.detach(),
                model.token_emb.weight.new_zeros(1),
            )
        ).view(model_cfg.vocab_size, model_cfg.d_model)
        effective_position = torch.cat(
            (
                model.pos_emb.rest.new_zeros(1, model_cfg.d_model),
                model.pos_emb.rest.detach(),
            ),
            dim=0,
        )
        virtual_token = (
            effective_token - initial_translation.unsqueeze(0)
        ).clone()
        virtual_position = (
            effective_position + initial_translation.unsqueeze(0)
        ).clone()

    embedding_params = [
        model.token_emb.weight,
        model.pos_emb.rest,
    ]
    embedding_m = [
        torch.zeros_like(virtual_token),
        torch.zeros_like(virtual_position),
    ]
    embedding_v = [
        torch.zeros_like(moment) for moment in embedding_m
    ]
    embedding_step = 0

    # Preserve ambient-coordinate AdamW dynamics for the three attention
    # weight gauges, attention bias, terminal bias, and six terminal columns.
    gauge_params = []
    for blk in model.blocks:
        gauge_params.extend(blk.attn.proj.weight_prefix)
        gauge_params.append(blk.attn.proj.bias)
        gauge_params.append(blk.mlp.fc2.bias)
        gauge_params.extend(blk.mlp.fc2.weight_prefix)
    excluded_ids = {
        id(p) for p in embedding_params + gauge_params
    }
    optimizer = torch.optim.AdamW(
        (
            p for p in model.parameters()
            if id(p) not in excluded_ids
        ),
        lr=train_cfg.lr,
        weight_decay=train_cfg.weight_decay,
    )
    gauge_m = [
        torch.zeros(p.numel() + 1, device=device, dtype=p.dtype)
        for p in gauge_params
    ]
    gauge_v = [torch.zeros_like(moment) for moment in gauge_m]
    gauge_step = 0
>>>>>>> REPLACE

<<<<<<< SEARCH
        optimizer.zero_grad(set_to_none=True)
        for gauge_param in gauge_params:
            gauge_param.grad = None
        loss.backward()

        full_gauge_grads = [
            model.token_emb.full_weight.grad.detach().reshape(-1),
            model.pos_emb.full_first.grad.detach(),
        ]
        for blk in model.blocks:
=======
        optimizer.zero_grad(set_to_none=True)
        for embedding_param in embedding_params:
            embedding_param.grad = None
        for gauge_param in gauge_params:
            gauge_param.grad = None
        loss.backward()

        embedding_grads = [
            model.token_emb.full_weight.grad.detach(),
            model.pos_emb.full_weight.grad.detach(),
        ]
        for embedding_param in embedding_params:
            embedding_param.grad = None

        full_gauge_grads = []
        for blk in model.blocks:
>>>>>>> REPLACE

<<<<<<< SEARCH
            for full_grad in full_gauge_grads:
                grad_sq = grad_sq + full_grad[-1].float().square()
=======
            for embedding_grad in embedding_grads:
                grad_sq = (
                    grad_sq
                    + embedding_grad.float().square().sum()
                )
            for full_grad in full_gauge_grads:
                grad_sq = grad_sq + full_grad[-1].float().square()
>>>>>>> REPLACE

<<<<<<< SEARCH
                blk.mlp.fc1.weight[:, -5:].copy_(
                    virtual_weight * virtual_scale.unsqueeze(0)
                )

        gauge_step += 1
=======
                blk.mlp.fc1.weight[:, -5:].copy_(
                    virtual_weight * virtual_scale.unsqueeze(0)
                )

        embedding_step += 1
        embedding_bc1 = 1.0 - 0.9 ** embedding_step
        embedding_bc2 = 1.0 - 0.999 ** embedding_step
        for virtual, ambient_grad, moment, variance in zip(
            (virtual_token, virtual_position),
            embedding_grads,
            embedding_m,
            embedding_v,
        ):
            grad = ambient_grad * clip_scale
            moment.mul_(0.9).add_(grad, alpha=0.1)
            variance.mul_(0.999).addcmul_(
                grad, grad, value=0.001
            )
            direction = (
                moment / embedding_bc1
            ) / (
                (variance / embedding_bc2).sqrt() + 1e-8
            )
            with torch.no_grad():
                virtual.mul_(
                    1.0 - lr_now * train_cfg.weight_decay
                )
                virtual.add_(direction, alpha=-lr_now)

        with torch.no_grad():
            # Restore the two already-qualified scalar gauges in the virtual
            # representative before applying the joint translation quotient.
            virtual_token.sub_(virtual_token[-1, -1])
            virtual_position[0].sub_(virtual_position[0, -1])

            translation = virtual_position[0].clone()
            effective_token = (
                virtual_token + translation.unsqueeze(0)
            )
            model.token_emb.weight.copy_(
                effective_token.reshape(-1)[:-1]
            )
            model.pos_emb.rest.copy_(
                virtual_position[1:] - translation.unsqueeze(0)
            )

        gauge_step += 1
>>>>>>> REPLACE