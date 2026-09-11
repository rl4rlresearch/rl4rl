MECHANISM: Two-position ambient-Adam gauge fixing

HYPOTHESIS: Gauge-fixing the first two positional embeddings while reproducing their full eight-coordinate AdamW moments and clipping will reduce the model to 1,626 parameters while retaining at least 99% accuracy.

INTENDED_EDIT: Extend the successful single-position gauge-fixed embedding and ambient optimizer from one positional row to two positional rows.

EVIDENCE: Single-position ambient-Adam gauge fixing achieved 100% accuracy with 1,627 parameters; extending it by one row is the smallest next reduction, while the 97.95% all-position result cautions against a larger jump.

<<<<<<< SEARCH
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
=======
class GaugeFixedPositionEmbedding(nn.Module):
    """Embedding with one shift-invariant scalar removed from two rows."""

    def __init__(self, num_embeddings: int, embedding_dim: int):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.fixed = nn.Parameter(torch.empty(2, embedding_dim - 1))
        self.rest = nn.Parameter(torch.empty(num_embeddings - 2, embedding_dim))
        self.full_fixed = None
        self.reset_parameters()

    @torch.no_grad()
    def reset_parameters(self, std: float = 1.0) -> None:
        # Draw the original full tensor so initialization and downstream RNG
        # consumption match nn.Embedding exactly.
        raw = self.fixed.new_empty(self.num_embeddings, self.embedding_dim)
        nn.init.normal_(raw, mean=0.0, std=std)
        self.fixed.copy_(raw[:2, :-1] - raw[:2, -1:])
        self.rest.copy_(raw[2:])

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        fixed = torch.cat((self.fixed, self.fixed.new_zeros(2, 1)), dim=-1)
        if torch.is_grad_enabled():
            fixed.retain_grad()
            self.full_fixed = fixed
        weight = torch.cat((fixed, self.rest), dim=0)
        return F.embedding(idx, weight)
>>>>>>> REPLACE

<<<<<<< SEARCH
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
=======
    # Each row's seven stored coordinates are differences from its omitted
    # eighth coordinate. Optimize all eight ambient coordinates, then apply
    # their update differences to the learned gauge-fixed parameters.
    gauge_param = model.pos_emb.fixed
    optimizer = torch.optim.AdamW(
        (p for p in model.parameters() if p is not gauge_param),
        lr=train_cfg.lr,
        weight_decay=train_cfg.weight_decay,
    )
    gauge_m = torch.zeros(
        2, model_cfg.d_model, device=device, dtype=gauge_param.dtype
    )
>>>>>>> REPLACE

<<<<<<< SEARCH
        full_gauge_grad = model.pos_emb.full_first.grad.detach()
=======
        full_gauge_grad = model.pos_emb.full_fixed.grad.detach()
>>>>>>> REPLACE

<<<<<<< SEARCH
            # Parameter gradients contain the first seven coordinates; add
            # the omitted ambient coordinate to reproduce full-model clipping.
            grad_sq = sum(
                p.grad.detach().float().square().sum()
                for p in model.parameters()
                if p.grad is not None
            )
            grad_sq = grad_sq + full_gauge_grad[-1].float().square()
=======
            # Parameter gradients contain each row's first seven coordinates;
            # add the omitted coordinates to reproduce full-model clipping.
            grad_sq = sum(
                p.grad.detach().float().square().sum()
                for p in model.parameters()
                if p.grad is not None
            )
            grad_sq = grad_sq + full_gauge_grad[:, -1].float().square().sum()
>>>>>>> REPLACE

<<<<<<< SEARCH
            gauge_param.add_(
                direction[-1] - direction[:-1], alpha=lr_now
            )
=======
            gauge_param.add_(
                direction[:, -1:] - direction[:, :-1], alpha=lr_now
            )
>>>>>>> REPLACE