MECHANISM: Mean-centered tied-embedding gauge with ambient AdamW

HYPOTHESIS: Representing the tied token embedding in a globally mean-zero gauge while preserving all 912 ambient AdamW moments will achieve at least 99% accuracy with 1,625 parameters, because centering avoids the large common offsets and numerical sensitivity of the prior anchor-coordinate gauge that reached 98.76%.

INTENDED_EDIT: Remove one exact global scalar from the tied token embedding, reuse one reconstructed weight tensor for lookup and logits, preserve initialization RNG order, and optimize its full ambient coordinates alongside the two qualified gauges.

EVIDENCE: The current dual-gauge model achieved 99.95% at 1,626 parameters, and the prior anchored tied-embedding gauge reached 98.76% at 1,625; its near miss motivates changing the gauge representative to the minimum-offset mean-zero form rather than removing another functional parameter.

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


class GaugeFixedPositionEmbedding(nn.Module):
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


class CenteredTokenEmbedding(nn.Module):
    """Tied token embedding with its global scalar gauge removed."""

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
            self.num_embeddings * self.embedding_dim
        )
        nn.init.normal_(raw, mean=0.0, std=std)
        raw.sub_(raw.mean())
        self.weight.copy_(raw[:-1])

    def forward(
        self, idx: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        last = -self.weight.sum().unsqueeze(0)
        full_weight = torch.cat((self.weight, last)).view(
            self.num_embeddings, self.embedding_dim
        )
        if torch.is_grad_enabled():
            full_weight.retain_grad()
        self.full_weight = full_weight
        return F.embedding(idx, full_weight), full_weight


class GaugeFixedPositionEmbedding(nn.Module):
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
        self.token_emb = CenteredTokenEmbedding(
            cfg.vocab_size, cfg.d_model
        )
        self.pos_emb = GaugeFixedPositionEmbedding(cfg.max_seq_len, cfg.d_model)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.ln_f = nn.LayerNorm(cfg.d_model)

        # Match the initialization draw consumed by the original tied head.
        head_scratch = torch.empty(cfg.vocab_size, cfg.d_model)
        nn.init.kaiming_uniform_(head_scratch, a=math.sqrt(5))

        self.apply(self._init_weights)

        # The tied head was initialized last in the original module traversal.
        self.token_emb.reset_parameters(std=0.02)
>>>>>>> REPLACE

<<<<<<< SEARCH
    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, GaugeFixedPositionEmbedding):
            module.reset_parameters(std=0.02)
        elif isinstance(module, GaugeFixedBiasLinear):
=======
    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, CenteredTokenEmbedding):
            module.reset_parameters(std=0.02)
        elif isinstance(module, GaugeFixedPositionEmbedding):
            module.reset_parameters(std=0.02)
        elif isinstance(module, GaugeFixedBiasLinear):
>>>>>>> REPLACE

<<<<<<< SEARCH
        pos = torch.arange(0, seqlen, device=idx.device).unsqueeze(0)
        x = self.token_emb(idx) + self.pos_emb(pos)
        x = self.drop(x)

        for blk in self.blocks:
            x = blk(x)

        x = self.ln_f(x)
        logits = self.lm_head(x)
=======
        pos = torch.arange(0, seqlen, device=idx.device).unsqueeze(0)
        token_x, token_weight = self.token_emb(idx)
        x = token_x + self.pos_emb(pos)
        x = self.drop(x)

        for blk in self.blocks:
            x = blk(x)

        x = self.ln_f(x)
        logits = F.linear(x, token_weight)
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
    # Optimize the two difference gauges and the centered tied embedding
    # through the AdamW moments of their full ambient vectors.
    difference_gauge_params = [model.pos_emb.first] + [
        blk.mlp.fc2.bias for blk in model.blocks
    ]
    token_gauge_param = model.token_emb.weight
    gauge_params = difference_gauge_params + [token_gauge_param]
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
        full_gauge_grads = (
            [model.pos_emb.full_first.grad.detach()]
            + [
                blk.mlp.fc2.full_bias.grad.detach()
                for blk in model.blocks
            ]
            + [model.token_emb.full_weight.grad.detach().reshape(-1)]
        )
        clip_scale = 1.0
        if train_cfg.grad_clip > 0:
            grad_sq = sum(
                p.grad.detach().float().square().sum()
                for p in model.parameters()
                if p.grad is not None and id(p) not in gauge_ids
            )
            for full_grad in full_gauge_grads:
                grad_sq = (
                    grad_sq
                    + full_grad.detach().float().square().sum()
                )
            total_norm = float(grad_sq.sqrt().item())
>>>>>>> REPLACE

<<<<<<< SEARCH
            with torch.no_grad():
                gauge_param.mul_(
                    1.0 - lr_now * train_cfg.weight_decay
                )
                gauge_param.add_(
                    direction[-1] - direction[:-1], alpha=lr_now
                )
=======
            with torch.no_grad():
                gauge_param.mul_(
                    1.0 - lr_now * train_cfg.weight_decay
                )
                if gauge_param is token_gauge_param:
                    gauge_param.add_(
                        direction.mean() - direction[:-1],
                        alpha=lr_now,
                    )
                else:
                    gauge_param.add_(
                        direction[-1] - direction[:-1],
                        alpha=lr_now,
                    )
>>>>>>> REPLACE