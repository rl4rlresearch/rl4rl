MECHANISM: Symmetric per-head value–projection ambient AdamW gauge

HYPOTHESIS: Removing the final value-bias coordinate from each attention head while transferring both omitted full-coordinate AdamW updates into the learned projection bias will produce a 1,624-parameter model with at least 99% accuracy.

INTENDED_EDIT: Reproduce the qualified positional and terminal-bias gauges, make `ln2` bias-free, and compact the value bias to three learned coordinates per head while preserving full eight-coordinate moments, clipping, weight decay, and projection-bias effects.

EVIDENCE: Reference Design 3 achieved 99.93% accuracy at 1,625 parameters after removing one value-bias coordinate with coupled ambient optimization; symmetrically removing the corresponding coordinate from the other head is the smallest direct extension of that successful mechanism.

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


class GaugeFixedPositionEmbedding(nn.Module):
    """Embedding with one shift-invariant positional scalar removed."""

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


class GaugeFixedBiasLinear(nn.Module):
    """Linear layer whose output bias omits its all-ones gauge scalar."""

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.empty(out_features - 1))
        self.full_bias = None
        self.reset_parameters()

    @torch.no_grad()
    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
        raw_bias = self.bias.new_empty(self.out_features)
        nn.init.uniform_(raw_bias, -bound, bound)
        self.bias.copy_(raw_bias[:-1] - raw_bias[-1])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        full_bias = torch.cat((self.bias, self.bias.new_zeros(1)))
        if torch.is_grad_enabled():
            full_bias.retain_grad()
            self.full_bias = full_bias
        return F.linear(x, self.weight, full_bias)


class CausalSelfAttention(nn.Module):
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.q_bias = nn.Parameter(torch.zeros(d_model))
        self.v_bias = nn.Parameter(torch.zeros(d_model))
        self.proj = nn.Linear(d_model, d_model)
=======
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.q_bias = nn.Parameter(torch.zeros(d_model))
        self.v_bias = nn.Parameter(
            torch.zeros(n_head, self.head_dim - 1)
        )
        self.full_v_bias = None
        self.proj = nn.Linear(d_model, d_model)
>>>>>>> REPLACE

<<<<<<< SEARCH
        q, k, v = qkv.chunk(3, dim=-1)
        q = q + self.q_bias
        v = v + self.v_bias

        q = q.view(bsz, seqlen, self.n_head, self.head_dim).transpose(1, 2)
=======
        q, k, v = qkv.chunk(3, dim=-1)
        q = q + self.q_bias
        full_v_bias = F.pad(self.v_bias, (0, 1)).reshape(-1)
        if torch.is_grad_enabled():
            full_v_bias.retain_grad()
            self.full_v_bias = full_v_bias
        v = v + full_v_bias

        q = q.view(bsz, seqlen, self.n_head, self.head_dim).transpose(1, 2)
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.drop = nn.Dropout(dropout)
=======
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = GaugeFixedBiasLinear(d_ff, d_model)
        self.drop = nn.Dropout(dropout)
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.ln2 = nn.LayerNorm(cfg.d_model)
=======
        self.ln2 = nn.LayerNorm(cfg.d_model, bias=False)
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.token_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_emb = nn.Embedding(cfg.max_seq_len, cfg.d_model)
        self.drop = nn.Dropout(cfg.dropout)
=======
        self.token_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_emb = GaugeFixedPositionEmbedding(
            cfg.max_seq_len, cfg.d_model
        )
        self.drop = nn.Dropout(cfg.dropout)
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
        if isinstance(module, GaugeFixedPositionEmbedding):
            module.reset_parameters(std=0.02)
        elif isinstance(module, GaugeFixedBiasLinear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
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
=======
    model = TinyDecoderLM(model_cfg).to(device)
    params = count_parameters(model)

    gauge_params = [model.pos_emb.first] + [
        blk.mlp.fc2.bias for blk in model.blocks
    ]
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
    value_m = [
        torch.zeros(model_cfg.d_model, device=device, dtype=p.dtype)
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
        for value_param, projection_param in zip(
            value_bias_params, projection_bias_params
        ):
            value_param.grad = None
            projection_param.grad = None
        loss.backward()

        full_gauge_grads = [model.pos_emb.full_first.grad.detach()] + [
            blk.mlp.fc2.full_bias.grad.detach() for blk in model.blocks
        ]
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
            for attention, full_grad in zip(
                value_attentions, full_value_grads
            ):
                omitted = full_grad.view(
                    attention.n_head, attention.head_dim
                )[:, -1]
                grad_sq = grad_sq + omitted.float().square().sum()
            total_norm = float(grad_sq.sqrt().item())
            clip_scale = min(
                1.0, train_cfg.grad_clip / (total_norm + 1e-6)
            )
            if clip_scale < 1.0:
                for p in model.parameters():
                    if p.grad is not None:
                        p.grad.mul_(clip_scale)

        optimizer.step()

        gauge_step += 1
        for gauge_param, full_grad, moment, variance in zip(
            gauge_params, full_gauge_grads, gauge_m, gauge_v
        ):
            ambient_grad = full_grad * clip_scale
            moment.mul_(0.9).add_(ambient_grad, alpha=0.1)
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
                    direction[-1] - direction[:-1], alpha=lr_now
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
            value_direction = value_direction.view(
                attention.n_head, attention.head_dim
            )
            with torch.no_grad():
                decay = 1.0 - lr_now * train_cfg.weight_decay
                value_param.mul_(decay)
                value_param.add_(
                    value_direction[:, :-1], alpha=-lr_now
                )
                projection_param.mul_(decay)
                projection_param.add_(
                    projection_direction, alpha=-lr_now
                )
                omitted_columns = attention.proj.weight[
                    :, attention.head_dim - 1 :: attention.head_dim
                ]
                projection_param.add_(
                    omitted_columns @ value_direction[:, -1],
                    alpha=-lr_now,
                )
>>>>>>> REPLACE