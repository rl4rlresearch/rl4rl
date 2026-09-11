MECHANISM: Ambient-Adam positional gauge fixing

HYPOTHESIS: Removing one LayerNorm-invariant scalar from the first positional embedding while reproducing its full eight-coordinate AdamW dynamics will yield 1,627 parameters and retain at least 99% accuracy.

INTENDED_EDIT: Apply the verified key-bias and `ln2`-bias removals, gauge-fix one positional coordinate, and update its seven learned differences using full-space AdamW moments and clipping.

EVIDENCE: The 1,628-parameter design reached 99.95%; naive single-position gauge fixing nearly passed at 98.59%, while ambient-coordinate optimization raised the harder all-position compression from 72.91% to 97.95%, indicating optimizer geometry is the remaining issue.

<<<<<<< SEARCH
    vocab_size: int


class CausalSelfAttention(nn.Module):
=======
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
        # Draw the original full tensor so initialization and downstream RNG
        # consumption match nn.Embedding exactly.
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


class CausalSelfAttention(nn.Module):
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)
=======
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.q_bias = nn.Parameter(torch.zeros(d_model))
        self.v_bias = nn.Parameter(torch.zeros(d_model))
        self.proj = nn.Linear(d_model, d_model)
>>>>>>> REPLACE

<<<<<<< SEARCH
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)

        q = q.view(bsz, seqlen, self.n_head, self.head_dim).transpose(1, 2)
=======
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)
        q = q + self.q_bias
        v = v + self.v_bias

        q = q.view(bsz, seqlen, self.n_head, self.head_dim).transpose(1, 2)
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.ln2 = nn.LayerNorm(cfg.d_model)
=======
        self.ln2 = nn.LayerNorm(cfg.d_model, bias=False)
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.pos_emb = nn.Embedding(cfg.max_seq_len, cfg.d_model)
=======
        self.pos_emb = GaugeFixedPositionEmbedding(cfg.max_seq_len, cfg.d_model)
>>>>>>> REPLACE

<<<<<<< SEARCH
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)
=======
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, GaugeFixedPositionEmbedding):
            module.reset_parameters(std=0.02)
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

    # The seven stored coordinates are differences from the omitted eighth
    # coordinate. Optimize all eight ambient coordinates, then apply their
    # update differences to the learned gauge-fixed parameter.
    gauge_param = model.pos_emb.first
    optimizer = torch.optim.AdamW(
        (p for p in model.parameters() if p is not gauge_param),
        lr=train_cfg.lr,
        weight_decay=train_cfg.weight_decay,
    )
    gauge_m = torch.zeros(
        model_cfg.d_model, device=device, dtype=gauge_param.dtype
    )
    gauge_v = torch.zeros_like(gauge_m)
    gauge_step = 0

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
        gauge_param.grad = None
        loss.backward()

        full_gauge_grad = model.pos_emb.full_first.grad.detach()
        clip_scale = 1.0
        if train_cfg.grad_clip > 0:
            # Parameter gradients contain the first seven coordinates; add
            # the omitted ambient coordinate to reproduce full-model clipping.
            grad_sq = sum(
                p.grad.detach().float().square().sum()
                for p in model.parameters()
                if p.grad is not None
            )
            grad_sq = grad_sq + full_gauge_grad[-1].float().square()
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
        ambient_grad = full_gauge_grad * clip_scale
        gauge_m.mul_(0.9).add_(ambient_grad, alpha=0.1)
        gauge_v.mul_(0.999).addcmul_(
            ambient_grad, ambient_grad, value=0.001
        )
        m_hat = gauge_m / (1.0 - 0.9 ** gauge_step)
        v_hat = gauge_v / (1.0 - 0.999 ** gauge_step)
        direction = m_hat / (v_hat.sqrt() + 1e-8)
        with torch.no_grad():
            gauge_param.mul_(1.0 - lr_now * train_cfg.weight_decay)
            gauge_param.add_(
                direction[-1] - direction[:-1], alpha=lr_now
            )
>>>>>>> REPLACE