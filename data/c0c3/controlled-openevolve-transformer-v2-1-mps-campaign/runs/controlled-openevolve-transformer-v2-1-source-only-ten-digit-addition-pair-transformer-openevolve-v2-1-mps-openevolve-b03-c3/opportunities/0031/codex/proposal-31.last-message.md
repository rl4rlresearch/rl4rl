MECHANISM: Ambient-Adam per-head relative-bias gauge fixing

HYPOTHESIS: Removing each relative-lag bias head’s softmax-invariant scalar while reproducing all 23-coordinate AdamW updates will retain at least 99% accuracy with 1,486 parameters.

INTENDED_EDIT: Port the qualified learned relative-lag design, compact each head’s 23 lag biases to 22 differences, and preserve its MLP-bias and coupled value–projection ambient optimizers.

EVIDENCE: Reference Design 3 achieved 99.98% accuracy with 1,488 parameters; unlike prior brittle functional approximations, the proposed two-parameter reduction removes exact per-head attention-softmax invariances while preserving full-coordinate initialization, clipping, moments, decay, and updates.

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
class GaugeFixedRelativePositionBias(nn.Module):
    """Per-head relative-lag bias with softmax-invariant shifts removed."""

    def __init__(self, n_head: int, max_seq_len: int, rng_width: int):
        super().__init__()
        self.n_head = n_head
        self.max_seq_len = max_seq_len
        self.rng_width = rng_width
        self.bias = nn.Parameter(
            torch.empty(n_head, max_seq_len - 1)
        )
        self.full_bias = None
        self.reset_parameters()

    @torch.no_grad()
    def reset_parameters(self, std: float = 1.0) -> None:
        # Consume the former absolute-position initialization draw so all
        # unchanged transformer parameters retain their initialization stream.
        raw = self.bias.new_empty(
            self.max_seq_len, self.rng_width
        )
        nn.init.normal_(raw, mean=0.0, std=std)
        ambient = raw.flatten()[: self.n_head * self.max_seq_len]
        ambient = ambient.view(self.n_head, self.max_seq_len)
        self.bias.copy_(
            ambient[:, :-1] - ambient[:, -1:]
        )

    def forward(self, seqlen: int) -> torch.Tensor:
        full_bias = torch.cat(
            (
                self.bias,
                self.bias.new_zeros(self.n_head, 1),
            ),
            dim=1,
        )
        if torch.is_grad_enabled():
            full_bias.retain_grad()
            self.full_bias = full_bias
        positions = torch.arange(
            seqlen, device=self.bias.device
        )
        distance = (
            positions[:, None] - positions[None, :]
        ).clamp_min(0)
        return full_bias[:, distance]
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.q_bias = nn.Parameter(torch.zeros(d_model))
        self.v_bias = nn.Parameter(torch.zeros(d_model))
        self.proj = nn.Linear(d_model, d_model)
=======
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.q_bias = nn.Parameter(torch.zeros(d_model))
        self.v_bias = nn.Parameter(torch.zeros(d_model - 1))
        self.full_v_bias = None
        self.proj = nn.Linear(d_model, d_model)
>>>>>>> REPLACE

<<<<<<< SEARCH
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz, seqlen, d_model = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)
        q = q + self.q_bias
        v = v + self.v_bias
=======
    def forward(
        self, x: torch.Tensor, position_bias: torch.Tensor
    ) -> torch.Tensor:
        bsz, seqlen, d_model = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)
        q = q + self.q_bias
        full_v_bias = torch.cat(
            (self.v_bias, self.v_bias.new_zeros(1))
        )
        if torch.is_grad_enabled():
            full_v_bias.retain_grad()
            self.full_v_bias = full_v_bias
        v = v + full_v_bias
>>>>>>> REPLACE

<<<<<<< SEARCH
        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        causal = self.mask[:seqlen, :seqlen]
=======
        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        att = att + position_bias.unsqueeze(0)
        causal = self.mask[:seqlen, :seqlen]
>>>>>>> REPLACE

<<<<<<< SEARCH
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x
=======
    def forward(
        self, x: torch.Tensor, position_bias: torch.Tensor
    ) -> torch.Tensor:
        x = x + self.attn(self.ln1(x), position_bias)
        x = x + self.mlp(self.ln2(x))
        return x
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.cfg = cfg
        self.token_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_emb = GaugeFixedPositionEmbedding(cfg.max_seq_len, cfg.d_model)
        self.drop = nn.Dropout(cfg.dropout)
=======
        self.cfg = cfg
        self.token_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_bias = GaugeFixedRelativePositionBias(
            cfg.n_head, cfg.max_seq_len, cfg.d_model
        )
        self.drop = nn.Dropout(cfg.dropout)
>>>>>>> REPLACE

<<<<<<< SEARCH
        if isinstance(module, GaugeFixedPositionEmbedding):
            module.reset_parameters(std=0.02)
=======
        if isinstance(module, GaugeFixedRelativePositionBias):
            module.reset_parameters(std=0.02)
>>>>>>> REPLACE

<<<<<<< SEARCH
        pos = torch.arange(0, seqlen, device=idx.device).unsqueeze(0)
        x = self.token_emb(idx) + self.pos_emb(pos)
        x = self.drop(x)

        for blk in self.blocks:
            x = blk(x)
=======
        x = self.drop(self.token_emb(idx))
        position_bias = self.pos_bias(seqlen)

        for blk in self.blocks:
            x = blk(x, position_bias)
>>>>>>> REPLACE

<<<<<<< SEARCH
    # Optimize each seven-coordinate gauge parameter through the AdamW
    # moments of its corresponding full eight-coordinate vector.
    gauge_params = [model.pos_emb.first] + [
        blk.mlp.fc2.bias for blk in model.blocks
    ]
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
    # Optimize compact gauges through their corresponding ambient-coordinate
    # AdamW moments, including coordinates omitted from learned storage.
    gauge_params = [
        blk.mlp.fc2.bias for blk in model.blocks
    ]
    position_bias_param = model.pos_bias.bias
    value_bias_params = [
        blk.attn.v_bias for blk in model.blocks
    ]
    projection_bias_params = [
        blk.attn.proj.bias for blk in model.blocks
    ]
    value_attentions = [
        blk.attn for blk in model.blocks
    ]
    special_ids = {
        id(p)
        for p in (
            gauge_params
            + [position_bias_param]
            + value_bias_params
            + projection_bias_params
        )
    }
    optimizer = torch.optim.AdamW(
        (p for p in model.parameters() if id(p) not in special_ids),
        lr=train_cfg.lr,
        weight_decay=train_cfg.weight_decay,
    )
    gauge_m = [
        torch.zeros(p.numel() + 1, device=device, dtype=p.dtype)
        for p in gauge_params
    ]
    gauge_v = [torch.zeros_like(moment) for moment in gauge_m]
    position_m = torch.zeros(
        model.pos_bias.n_head,
        model.pos_bias.max_seq_len,
        device=device,
        dtype=position_bias_param.dtype,
    )
    position_v = torch.zeros_like(position_m)
    value_m = [
        torch.zeros(p.numel() + 1, device=device, dtype=p.dtype)
        for p in value_bias_params
    ]
    value_v = [torch.zeros_like(moment) for moment in value_m]
    projection_m = [
        torch.zeros_like(p) for p in projection_bias_params
    ]
    projection_v = [
        torch.zeros_like(moment) for moment in projection_m
    ]
    gauge_step = 0
>>>>>>> REPLACE

<<<<<<< SEARCH
        optimizer.zero_grad(set_to_none=True)
        for gauge_param in gauge_params:
            gauge_param.grad = None
        loss.backward()

        full_gauge_grads = [model.pos_emb.full_first.grad.detach()] + [
            blk.mlp.fc2.full_bias.grad.detach() for blk in model.blocks
        ]
        clip_scale = 1.0
        if train_cfg.grad_clip > 0:
            grad_sq = sum(
                p.grad.detach().float().square().sum()
                for p in model.parameters()
                if p.grad is not None
            )
            for full_grad in full_gauge_grads:
                grad_sq = grad_sq + full_grad[-1].float().square()
            total_norm = float(grad_sq.sqrt().item())
            clip_scale = min(
                1.0, train_cfg.grad_clip / (total_norm + 1e-6)
            )
            if clip_scale < 1.0:
                for p in model.parameters():
                    if p.grad is not None:
                        p.grad.mul_(clip_scale)

        optimizer.step()
=======
        optimizer.zero_grad(set_to_none=True)
        for gauge_param in gauge_params:
            gauge_param.grad = None
        position_bias_param.grad = None
        for value_param, projection_param in zip(
            value_bias_params, projection_bias_params
        ):
            value_param.grad = None
            projection_param.grad = None
        loss.backward()

        full_gauge_grads = [
            blk.mlp.fc2.full_bias.grad.detach() for blk in model.blocks
        ]
        full_position_grad = model.pos_bias.full_bias.grad.detach()
        full_value_grads = [
            attention.full_v_bias.grad.detach()
            for attention in value_attentions
        ]
        projection_grads = [
            p.grad.detach().clone() for p in projection_bias_params
        ]
        clip_scale = 1.0
        if train_cfg.grad_clip > 0:
            grad_sq = sum(
                p.grad.detach().float().square().sum()
                for p in model.parameters()
                if p.grad is not None
            )
            for full_grad in full_gauge_grads:
                grad_sq = grad_sq + full_grad[-1].float().square()
            grad_sq = (
                grad_sq
                + full_position_grad[:, -1]
                .float()
                .square()
                .sum()
            )
            for full_grad in full_value_grads:
                grad_sq = grad_sq + full_grad[-1].float().square()
            total_norm = float(grad_sq.sqrt().item())
            clip_scale = min(
                1.0, train_cfg.grad_clip / (total_norm + 1e-6)
            )
            if clip_scale < 1.0:
                for p in model.parameters():
                    if p.grad is not None:
                        p.grad.mul_(clip_scale)

        optimizer.step()
>>>>>>> REPLACE

<<<<<<< SEARCH
                gauge_param.add_(
                    direction[-1] - direction[:-1], alpha=lr_now
                )

        if (step % train_cfg.eval_interval == 0) or (step == train_cfg.train_steps - 1):
=======
                gauge_param.add_(
                    direction[-1] - direction[:-1], alpha=lr_now
                )

        position_grad = full_position_grad * clip_scale
        position_m.mul_(0.9).add_(position_grad, alpha=0.1)
        position_v.mul_(0.999).addcmul_(
            position_grad, position_grad, value=0.001
        )
        position_direction = (
            position_m / (1.0 - 0.9 ** gauge_step)
        ) / (
            (
                position_v / (1.0 - 0.999 ** gauge_step)
            ).sqrt()
            + 1e-8
        )
        with torch.no_grad():
            position_bias_param.mul_(
                1.0 - lr_now * train_cfg.weight_decay
            )
            position_bias_param.add_(
                position_direction[:, -1:]
                - position_direction[:, :-1],
                alpha=lr_now,
            )

        for (
            value_param,
            projection_param,
            attention,
            full_grad,
            projection_grad,
            value_moment,
            value_variance,
            projection_moment,
            projection_variance,
        ) in zip(
            value_bias_params,
            projection_bias_params,
            value_attentions,
            full_value_grads,
            projection_grads,
            value_m,
            value_v,
            projection_m,
            projection_v,
        ):
            value_grad = full_grad * clip_scale
            projection_grad = projection_grad * clip_scale
            value_moment.mul_(0.9).add_(value_grad, alpha=0.1)
            value_variance.mul_(0.999).addcmul_(
                value_grad, value_grad, value=0.001
            )
            projection_moment.mul_(0.9).add_(
                projection_grad, alpha=0.1
            )
            projection_variance.mul_(0.999).addcmul_(
                projection_grad, projection_grad, value=0.001
            )
            value_direction = (
                value_moment / (1.0 - 0.9 ** gauge_step)
            ) / (
                (
                    value_variance
                    / (1.0 - 0.999 ** gauge_step)
                ).sqrt()
                + 1e-8
            )
            projection_direction = (
                projection_moment / (1.0 - 0.9 ** gauge_step)
            ) / (
                (
                    projection_variance
                    / (1.0 - 0.999 ** gauge_step)
                ).sqrt()
                + 1e-8
            )
            with torch.no_grad():
                decay = 1.0 - lr_now * train_cfg.weight_decay
                value_param.mul_(decay)
                value_param.add_(
                    value_direction[:-1], alpha=-lr_now
                )
                projection_param.mul_(decay)
                projection_param.add_(
                    projection_direction, alpha=-lr_now
                )
                projection_param.add_(
                    attention.proj.weight[:, -1]
                    * value_direction[-1],
                    alpha=-lr_now,
                )

        if (step % train_cfg.eval_interval == 0) or (step == train_cfg.train_steps - 1):
>>>>>>> REPLACE