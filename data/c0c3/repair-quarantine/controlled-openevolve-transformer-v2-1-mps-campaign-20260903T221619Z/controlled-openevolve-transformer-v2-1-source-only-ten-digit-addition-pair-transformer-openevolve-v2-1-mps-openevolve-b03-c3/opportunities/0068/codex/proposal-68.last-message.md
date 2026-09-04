MECHANISM: Complete `ln2` scale absorption with ambient AdamW

HYPOTHESIS: A 1,527-parameter model will retain at least 99% accuracy because the verified seven-scale design achieved 99.93%, while the eighth and final `ln2` scale is equally absorbable into its corresponding `fc1` column without reducing model expressivity.

INTENDED_EDIT: Reproduce the qualified tied-embedding, positional, six-terminal-column, and three-attention-column gauges, then absorb all eight `ln2` scales into `fc1`; use a parameter-free LayerNorm instead of creating a forbidden zero-length parameter, and preserve full weight/scale AdamW dynamics in optimizer state.

EVIDENCE: Reference Design 1 achieved 99.93% accuracy with 1,528 parameters after absorbing seven of eight scales; this tests the smallest remaining extension of that proven exact factorization.

<<<<<<< SEARCH
@dataclass
class ModelConfig:
    n_layer: int
    d_model: int
    n_head: int
    d_ff: int
    dropout: float
    max_seq_len: int
    vocab_size: int


class CausalSelfAttention(nn.Module):
=======
@dataclass
class ModelConfig:
    n_layer: int
    d_model: int
    n_head: int
    d_ff: int
    dropout: float
    max_seq_len: int
    vocab_size: int


class GaugeFixedTokenEmbedding(nn.Module):
    """Tied embedding with its global scalar-shift gauge removed."""

    def __init__(self, num_embeddings: int, embedding_dim: int):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.weight = nn.Parameter(
            torch.empty(num_embeddings * embedding_dim - 1)
        )
        self.full_weight = None
        self.reset_parameters()

    @torch.no_grad()
    def reset_parameters(self, std: float = 1.0) -> None:
        raw = self.weight.new_empty(
            self.num_embeddings, self.embedding_dim
        )
        nn.init.normal_(raw, mean=0.0, std=std)
        flat = raw.reshape(-1)
        self.weight.copy_(flat[:-1] - flat[-1])

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        full_weight = torch.cat(
            (self.weight, self.weight.new_zeros(1))
        ).view(self.num_embeddings, self.embedding_dim)
        if torch.is_grad_enabled():
            full_weight.retain_grad()
        self.full_weight = full_weight
        return F.embedding(idx, full_weight)

    def project(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.full_weight)


class TiedTokenProjection(nn.Module):
    """Parameter-free output view of the learned token embedding."""

    def __init__(self, embedding: GaugeFixedTokenEmbedding):
        super().__init__()
        self.in_features = embedding.embedding_dim
        self.out_features = embedding.num_embeddings
        object.__setattr__(self, "embedding", embedding)

        scratch = torch.empty(self.out_features, self.in_features)
        nn.init.kaiming_uniform_(scratch, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.embedding.project(x)


class GaugeFixedPositionEmbedding(nn.Module):
    """Embedding with one shift-invariant positional scalar removed."""

    def __init__(self, num_embeddings: int, embedding_dim: int):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.first = nn.Parameter(torch.empty(embedding_dim - 1))
        self.rest = nn.Parameter(
            torch.empty(num_embeddings - 1, embedding_dim)
        )
        self.full_first = None
        self.reset_parameters()

    @torch.no_grad()
    def reset_parameters(self, std: float = 1.0) -> None:
        raw = self.first.new_empty(
            self.num_embeddings, self.embedding_dim
        )
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


class GaugeFixedTerminalLinear(nn.Module):
    """Linear layer with bias and six output-shift column gauges removed."""

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight_prefix = nn.ParameterList(
            [
                nn.Parameter(torch.empty(out_features - 1))
                for _ in range(6)
            ]
        )
        self.weight_rest = nn.Parameter(
            torch.empty(out_features, in_features - 6)
        )
        self.bias = nn.Parameter(torch.empty(out_features - 1))
        self.full_weight_prefix = None
        self.full_bias = None
        self.reset_parameters()

    @torch.no_grad()
    def reset_parameters(self) -> None:
        raw_weight = self.weight_rest.new_empty(
            self.out_features, self.in_features
        )
        nn.init.kaiming_uniform_(raw_weight, a=math.sqrt(5))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(raw_weight)
        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
        raw_bias = self.bias.new_empty(self.out_features)
        nn.init.uniform_(raw_bias, -bound, bound)
        for column, stored in enumerate(self.weight_prefix):
            stored.copy_(
                raw_weight[:-1, column] - raw_weight[-1, column]
            )
        self.weight_rest.copy_(raw_weight[:, 6:])
        self.bias.copy_(raw_bias[:-1] - raw_bias[-1])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        full_weight_prefix = [
            torch.cat((stored, stored.new_zeros(1)))
            for stored in self.weight_prefix
        ]
        full_bias = torch.cat((self.bias, self.bias.new_zeros(1)))
        if torch.is_grad_enabled():
            for full_weight in full_weight_prefix:
                full_weight.retain_grad()
            full_bias.retain_grad()
            self.full_weight_prefix = full_weight_prefix
            self.full_bias = full_bias
        weight = torch.cat(
            (
                torch.stack(full_weight_prefix, dim=1),
                self.weight_rest,
            ),
            dim=1,
        )
        return F.linear(x, weight, full_bias)


class GaugeFixedAttentionProjection(nn.Module):
    """Linear projection with bias and three output-shift gauges removed."""

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight_prefix = nn.ParameterList(
            [
                nn.Parameter(torch.empty(out_features - 1))
                for _ in range(3)
            ]
        )
        self.weight_rest = nn.Parameter(
            torch.empty(out_features, in_features - 3)
        )
        self.bias = nn.Parameter(torch.empty(out_features - 1))
        self.full_weight_prefix = None
        self.full_bias = None
        self.reset_parameters()

    @torch.no_grad()
    def reset_parameters(self) -> None:
        raw_weight = self.weight_rest.new_empty(
            self.out_features, self.in_features
        )
        nn.init.kaiming_uniform_(raw_weight, a=math.sqrt(5))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(raw_weight)
        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
        raw_bias = self.bias.new_empty(self.out_features)
        nn.init.uniform_(raw_bias, -bound, bound)
        for column, stored in enumerate(self.weight_prefix):
            stored.copy_(
                raw_weight[:-1, column] - raw_weight[-1, column]
            )
        self.weight_rest.copy_(raw_weight[:, 3:])
        self.bias.copy_(raw_bias[:-1] - raw_bias[-1])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        full_weight_prefix = [
            torch.cat((stored, stored.new_zeros(1)))
            for stored in self.weight_prefix
        ]
        full_bias = torch.cat((self.bias, self.bias.new_zeros(1)))
        if torch.is_grad_enabled():
            for full_weight in full_weight_prefix:
                full_weight.retain_grad()
            full_bias.retain_grad()
            self.full_weight_prefix = full_weight_prefix
            self.full_bias = full_bias
        weight = torch.cat(
            (
                torch.stack(full_weight_prefix, dim=1),
                self.weight_rest,
            ),
            dim=1,
        )
        return F.linear(x, weight, full_bias)


class FullyAbsorbedScaleLayerNorm(nn.Module):
    """Parameter-free LayerNorm whose scales live in following fc1 columns."""

    def __init__(self, normalized_shape: int, eps: float = 1e-5):
        super().__init__()
        self.normalized_shape = normalized_shape
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.layer_norm(
            x, (self.normalized_shape,), None, None, self.eps
        )


class CausalSelfAttention(nn.Module):
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.n_head = n_head
        self.head_dim = d_model // n_head
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)
        self.attn_drop = nn.Dropout(dropout)
=======
        self.n_head = n_head
        self.head_dim = d_model // n_head
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, self.head_dim, bias=False)
        self.v_proj = nn.Linear(d_model, self.head_dim, bias=False)
        self.q_bias = nn.Parameter(torch.zeros(d_model))
        self.proj = GaugeFixedAttentionProjection(d_model, d_model)
        self.attn_drop = nn.Dropout(dropout)
>>>>>>> REPLACE

<<<<<<< SEARCH
        bsz, seqlen, d_model = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)

        q = q.view(bsz, seqlen, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(bsz, seqlen, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(bsz, seqlen, self.n_head, self.head_dim).transpose(1, 2)
=======
        bsz, seqlen, d_model = x.shape
        q = self.q_proj(x) + self.q_bias
        k = self.k_proj(x)
        v = self.v_proj(x)

        q = q.view(
            bsz, seqlen, self.n_head, self.head_dim
        ).transpose(1, 2)
        k = k.unsqueeze(1)
        v = v.unsqueeze(1)
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.fc2 = nn.Linear(d_ff, d_model)
=======
        self.fc2 = GaugeFixedTerminalLinear(d_ff, d_model)
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.attn = CausalSelfAttention(cfg.d_model, cfg.n_head, cfg.dropout, cfg.max_seq_len)
        self.ln2 = nn.LayerNorm(cfg.d_model)
=======
        self.ln1 = nn.LayerNorm(cfg.d_model, bias=False)
        self.attn = CausalSelfAttention(cfg.d_model, cfg.n_head, cfg.dropout, cfg.max_seq_len)
        self.ln2 = FullyAbsorbedScaleLayerNorm(cfg.d_model)
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.token_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_emb = nn.Embedding(cfg.max_seq_len, cfg.d_model)
=======
        self.token_emb = GaugeFixedTokenEmbedding(
            cfg.vocab_size, cfg.d_model
        )
        self.pos_emb = GaugeFixedPositionEmbedding(
            cfg.max_seq_len, cfg.d_model
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
        # Weight tying with input embeddings.
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.token_emb.weight
=======
        # Parameter-free output view preserves input/output weight tying.
        self.lm_head = TiedTokenProjection(self.token_emb)
>>>>>>> REPLACE

<<<<<<< SEARCH
    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)
=======
    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, GaugeFixedTokenEmbedding):
            module.reset_parameters(std=0.02)
        elif isinstance(module, TiedTokenProjection):
            module.embedding.reset_parameters(std=0.02)
        elif isinstance(module, GaugeFixedPositionEmbedding):
            module.reset_parameters(std=0.02)
        elif isinstance(module, GaugeFixedAttentionProjection):
            with torch.no_grad():
                raw_weight = module.weight_rest.new_empty(
                    module.out_features, module.in_features
                )
                nn.init.normal_(raw_weight, mean=0.0, std=0.02)
                for column, stored in enumerate(module.weight_prefix):
                    stored.copy_(
                        raw_weight[:-1, column]
                        - raw_weight[-1, column]
                    )
                module.weight_rest.copy_(raw_weight[:, 3:])
                nn.init.zeros_(module.bias)
        elif isinstance(module, GaugeFixedTerminalLinear):
            with torch.no_grad():
                raw_weight = module.weight_rest.new_empty(
                    module.out_features, module.in_features
                )
                nn.init.normal_(raw_weight, mean=0.0, std=0.02)
                for column, stored in enumerate(module.weight_prefix):
                    stored.copy_(
                        raw_weight[:-1, column]
                        - raw_weight[-1, column]
                    )
                module.weight_rest.copy_(raw_weight[:, 6:])
                nn.init.zeros_(module.bias)
        elif isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)
>>>>>>> REPLACE

<<<<<<< SEARCH
    model = TinyDecoderLM(model_cfg).to(device)
    params = count_parameters(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=train_cfg.lr, weight_decay=train_cfg.weight_decay)

    sampler = TrainBatchSampler(train_cfg.batch_size, train_cfg.seed + 1337, reserved_hashes)
=======
    model = TinyDecoderLM(model_cfg).to(device)
    params = count_parameters(model)

    # Optimize omitted gauge coordinates in their full ambient spaces.
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

    # All eight ln2 scales are represented only as optimizer-coordinate
    # state; fc1 stores and uses their products with the ambient weights.
    absorbed_scales = [
        torch.ones(8, device=device, dtype=blk.mlp.fc1.weight.dtype)
        for blk in model.blocks
    ]
    absorbed_weight_m = [
        torch.zeros_like(blk.mlp.fc1.weight[:, -8:])
        for blk in model.blocks
    ]
    absorbed_weight_v = [
        torch.zeros_like(moment) for moment in absorbed_weight_m
    ]
    absorbed_scale_m = [
        torch.zeros_like(scale) for scale in absorbed_scales
    ]
    absorbed_scale_v = [
        torch.zeros_like(moment) for moment in absorbed_scale_m
    ]
    absorbed_step = 0

    sampler = TrainBatchSampler(train_cfg.batch_size, train_cfg.seed + 1337, reserved_hashes)
>>>>>>> REPLACE

<<<<<<< SEARCH
        _, loss = model(x, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if train_cfg.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg.grad_clip)
        optimizer.step()
=======
        _, loss = model(x, y)
        optimizer.zero_grad(set_to_none=True)
        for gauge_param in gauge_params:
            gauge_param.grad = None
        loss.backward()

        full_gauge_grads = [
            model.token_emb.full_weight.grad.detach().reshape(-1),
            model.pos_emb.full_first.grad.detach(),
        ]
        for blk in model.blocks:
            full_gauge_grads.extend(
                full_weight.grad.detach()
                for full_weight in blk.attn.proj.full_weight_prefix
            )
            full_gauge_grads.append(
                blk.attn.proj.full_bias.grad.detach()
            )
            full_gauge_grads.append(
                blk.mlp.fc2.full_bias.grad.detach()
            )
            full_gauge_grads.extend(
                full_weight.grad.detach()
                for full_weight in blk.mlp.fc2.full_weight_prefix
            )

        absorbed_grads = []
        for blk, virtual_scale in zip(model.blocks, absorbed_scales):
            effective_grad = (
                blk.mlp.fc1.weight.grad[:, -8:].detach().clone()
            )
            virtual_weight = (
                blk.mlp.fc1.weight[:, -8:].detach()
                / virtual_scale.unsqueeze(0)
            )
            ambient_weight_grad = (
                effective_grad * virtual_scale.unsqueeze(0)
            )
            ambient_scale_grad = (
                effective_grad * virtual_weight
            ).sum(dim=0)
            absorbed_grads.append(
                (
                    effective_grad,
                    virtual_weight,
                    ambient_weight_grad,
                    ambient_scale_grad,
                )
            )

        clip_scale = 1.0
        if train_cfg.grad_clip > 0:
            grad_sq = sum(
                p.grad.detach().float().square().sum()
                for p in model.parameters()
                if p.grad is not None
            )
            for full_grad in full_gauge_grads:
                grad_sq = (
                    grad_sq
                    + full_grad[-1].float().square()
                )
            for (
                effective_grad,
                _,
                ambient_weight_grad,
                ambient_scale_grad,
            ) in absorbed_grads:
                grad_sq = (
                    grad_sq
                    - effective_grad.float().square().sum()
                    + ambient_weight_grad.float().square().sum()
                    + ambient_scale_grad.float().square().sum()
                )
            total_norm = float(grad_sq.sqrt().item())
            clip_scale = min(
                1.0, train_cfg.grad_clip / (total_norm + 1e-6)
            )
            if clip_scale < 1.0:
                for p in model.parameters():
                    if p.grad is not None:
                        p.grad.mul_(clip_scale)

        for blk in model.blocks:
            blk.mlp.fc1.weight.grad[:, -8:].zero_()

        optimizer.step()

        absorbed_step += 1
        absorbed_bc1 = 1.0 - 0.9 ** absorbed_step
        absorbed_bc2 = 1.0 - 0.999 ** absorbed_step
        for i, (blk, absorbed_grad) in enumerate(
            zip(model.blocks, absorbed_grads)
        ):
            (
                _,
                virtual_weight,
                ambient_weight_grad,
                ambient_scale_grad,
            ) = absorbed_grad
            weight_grad = ambient_weight_grad * clip_scale
            scale_grad = ambient_scale_grad * clip_scale
            weight_moment = absorbed_weight_m[i]
            weight_variance = absorbed_weight_v[i]
            scale_moment = absorbed_scale_m[i]
            scale_variance = absorbed_scale_v[i]
            virtual_scale = absorbed_scales[i]

            weight_moment.mul_(0.9).add_(
                weight_grad, alpha=0.1
            )
            weight_variance.mul_(0.999).addcmul_(
                weight_grad, weight_grad, value=0.001
            )
            scale_moment.mul_(0.9).add_(
                scale_grad, alpha=0.1
            )
            scale_variance.mul_(0.999).addcmul_(
                scale_grad, scale_grad, value=0.001
            )
            weight_direction = (
                weight_moment / absorbed_bc1
            ) / (
                (weight_variance / absorbed_bc2).sqrt() + 1e-8
            )
            scale_direction = (
                scale_moment / absorbed_bc1
            ) / (
                (scale_variance / absorbed_bc2).sqrt() + 1e-8
            )

            with torch.no_grad():
                decay = 1.0 - lr_now * train_cfg.weight_decay
                virtual_weight.mul_(decay).add_(
                    weight_direction, alpha=-lr_now
                )
                virtual_scale.mul_(decay).add_(
                    scale_direction, alpha=-lr_now
                )
                blk.mlp.fc1.weight[:, -8:].copy_(
                    virtual_weight
                    * virtual_scale.unsqueeze(0)
                )

        gauge_step += 1
        for gauge_param, full_grad, moment, variance in zip(
            gauge_params, full_gauge_grads, gauge_m, gauge_v
        ):
            ambient_grad = full_grad * clip_scale
            moment.mul_(0.9).add_(
                ambient_grad, alpha=0.1
            )
            variance.mul_(0.999).addcmul_(
                ambient_grad, ambient_grad, value=0.001
            )
            m_hat = moment / (1.0 - 0.9 ** gauge_step)
            v_hat = variance / (1.0 - 0.999 ** gauge_step)
            direction = m_hat / (v_hat.sqrt() + 1e-8)
            with torch.no_grad():
                gauge_param.mul_(
                    1.0 - lr_now * train_cfg.weight_decay
                )
                gauge_param.add_(
                    direction[-1] - direction[:-1],
                    alpha=lr_now,
                )
>>>>>>> REPLACE