MECHANISM: Symmetric mean-zero tied-embedding gauge with ambient AdamW

HYPOTHESIS: Centering the tied embedding in the mean-zero gauge and reproducing its full 912-coordinate AdamW update will achieve at least 99% accuracy with 1,625 parameters, because it removes the same exact global-shift invariance as the 98.76% near-miss while avoiding an asymmetric fixed-coordinate representation.

INTENDED_EDIT: Replace the tied token embedding with 911 learned mean-zero coordinates, preserve the original initialization RNG sequence, use the reconstructed weight for lookup and logits, and optimize it through full ambient moments and clipping alongside the two qualified gauges.

EVIDENCE: The qualified positional-plus-terminal-bias design achieved 99.95% with 1,626 parameters, and the prior global tied-embedding gauge reached 98.76% with 1,625—far closer to the threshold than the other one-parameter extensions—making a numerically symmetric parameterization the most informative next test.

<<<<<<< SEARCH
class GaugeFixedPositionEmbedding(nn.Module):
    """Position embedding with one shift-invariant scalar removed."""
=======
class GaugeFixedTokenEmbedding(nn.Module):
    """Tied embedding constrained to the mean-zero global-shift gauge."""

    def __init__(self, num_embeddings: int, embedding_dim: int):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.coordinates = nn.Parameter(
            torch.empty(num_embeddings * embedding_dim - 1)
        )
        self.full_weight = None
        self.reset_parameters()

    @torch.no_grad()
    def reset_parameters(self, std: float = 1.0) -> None:
        raw = self.coordinates.new_empty(
            self.num_embeddings, self.embedding_dim
        )
        nn.init.normal_(raw, mean=0.0, std=std)
        centered = raw.reshape(-1) - raw.mean()
        self.coordinates.copy_(centered[:-1])

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        last = -self.coordinates.sum()
        weight = torch.cat(
            (self.coordinates, last.unsqueeze(0))
        ).view(self.num_embeddings, self.embedding_dim)
        if torch.is_grad_enabled():
            weight.retain_grad()
        self.full_weight = weight
        return F.embedding(idx, weight)


class TiedLMHead(nn.Module):
    """Parameter-free projection using a reconstructed tied embedding."""

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # Match the RNG consumption of the replaced bias-free Linear.
        scratch = torch.empty(out_features, in_features)
        nn.init.kaiming_uniform_(scratch, a=math.sqrt(5))

    def forward(
        self, x: torch.Tensor, weight: torch.Tensor
    ) -> torch.Tensor:
        return F.linear(x, weight)


class GaugeFixedPositionEmbedding(nn.Module):
    """Position embedding with one shift-invariant scalar removed."""
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.cfg = cfg
        self.token_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_emb = GaugeFixedPositionEmbedding(cfg.max_seq_len, cfg.d_model)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.ln_f = nn.LayerNorm(cfg.d_model)

        # Weight tying with input embeddings.
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.token_emb.weight

        self.apply(self._init_weights)
=======
        self.cfg = cfg
        self.token_emb = GaugeFixedTokenEmbedding(
            cfg.vocab_size, cfg.d_model
        )
        self.pos_emb = GaugeFixedPositionEmbedding(cfg.max_seq_len, cfg.d_model)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.ln_f = nn.LayerNorm(cfg.d_model)

        # The head reuses the reconstructed embedding without another parameter.
        self.lm_head = TiedLMHead(cfg.d_model, cfg.vocab_size)

        self.apply(self._init_weights)
        # In the original tied implementation, visiting lm_head during apply
        # initializes the shared token weight a second time.
        self.token_emb.reset_parameters(std=0.02)
>>>>>>> REPLACE

<<<<<<< SEARCH
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, GaugeFixedPositionEmbedding):
            module.reset_parameters(std=0.02)
=======
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, GaugeFixedTokenEmbedding):
            module.reset_parameters(std=0.02)
        elif isinstance(module, GaugeFixedPositionEmbedding):
            module.reset_parameters(std=0.02)
>>>>>>> REPLACE

<<<<<<< SEARCH
        logits = self.lm_head(x)
=======
        logits = self.lm_head(x, self.token_emb.full_weight)
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
    # Optimize the two anchored gauges and the symmetric tied-embedding
    # gauge through moments in their original ambient coordinate spaces.
    anchored_gauge_params = [model.pos_emb.first] + [
        blk.mlp.fc2.bias for blk in model.blocks
    ]
    embedding_gauge_param = model.token_emb.coordinates
    gauge_params = anchored_gauge_params + [embedding_gauge_param]
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
>>>>>>> REPLACE

<<<<<<< SEARCH
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
=======
        full_gauge_grads = [model.pos_emb.full_first.grad.detach()] + [
            blk.mlp.fc2.full_bias.grad.detach() for blk in model.blocks
        ] + [
            model.token_emb.full_weight.grad.detach().reshape(-1)
        ]
        clip_scale = 1.0
        if train_cfg.grad_clip > 0:
            # Begin with the tied embedding, which was the first parameter in
            # the ungauged model, then preserve the original accumulation order.
            grad_sq = full_gauge_grads[-1].float().square().sum()
            for p in model.parameters():
                if (
                    p.grad is not None
                    and p is not embedding_gauge_param
                ):
                    grad_sq = (
                        grad_sq
                        + p.grad.detach().float().square().sum()
                    )
            for full_grad in full_gauge_grads[:-1]:
                grad_sq = grad_sq + full_grad[-1].float().square()
            total_norm = float(grad_sq.sqrt().item())
>>>>>>> REPLACE

<<<<<<< SEARCH
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
=======
        gauge_step += 1
        for gauge_param, full_grad, moment, variance in zip(
            anchored_gauge_params,
            full_gauge_grads[:-1],
            gauge_m[:-1],
            gauge_v[:-1],
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

        embedding_grad = full_gauge_grads[-1] * clip_scale
        embedding_m = gauge_m[-1]
        embedding_v = gauge_v[-1]
        embedding_m.mul_(0.9).add_(embedding_grad, alpha=0.1)
        embedding_v.mul_(0.999).addcmul_(
            embedding_grad, embedding_grad, value=0.001
        )
        embedding_m_hat = embedding_m / (1.0 - 0.9 ** gauge_step)
        embedding_v_hat = embedding_v / (1.0 - 0.999 ** gauge_step)
        embedding_direction = embedding_m_hat / (
            embedding_v_hat.sqrt() + 1e-8
        )
        with torch.no_grad():
            embedding_gauge_param.mul_(
                1.0 - lr_now * train_cfg.weight_decay
            )
            embedding_gauge_param.add_(
                embedding_direction.mean()
                - embedding_direction[:-1],
                alpha=lr_now,
            )
>>>>>>> REPLACE