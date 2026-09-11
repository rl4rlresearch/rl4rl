MECHANISM: Second positional-row shift gauge atop five-scale factorization

HYPOTHESIS: A 1,534-parameter model will retain at least 99% accuracy because the qualified five-scale design already achieved 99.95%, while the scalar component removed from a second positional vector is independently erased by the pre-norm and final LayerNorm operations and its full AdamW dynamics are preserved in ambient coordinates.

INTENDED_EDIT: Extend the qualified five-scale `ln2` absorption to the current design and gauge-fix the second positional embedding row, including its ambient optimizer moments, gradients, clipping contribution, and update.

EVIDENCE: Reference Design 2 achieved 99.95% accuracy with 1,535 parameters using five absorbed `ln2` scales, and every qualified reduced design already uses the identical positional shift gauge for the first row; applying that exact invariance to one additional row is the smallest new reduction atop the best qualified design.

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
    """Embedding with two independent positional shift gauges removed."""

    def __init__(self, num_embeddings: int, embedding_dim: int):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.first = nn.Parameter(torch.empty(embedding_dim - 1))
        self.second = nn.Parameter(torch.empty(embedding_dim - 1))
        self.rest = nn.Parameter(torch.empty(num_embeddings - 2, embedding_dim))
        self.full_first = None
        self.full_second = None
        self.reset_parameters()

    @torch.no_grad()
    def reset_parameters(self, std: float = 1.0) -> None:
        raw = self.first.new_empty(self.num_embeddings, self.embedding_dim)
        nn.init.normal_(raw, mean=0.0, std=std)
        self.first.copy_(raw[0, :-1] - raw[0, -1])
        self.second.copy_(raw[1, :-1] - raw[1, -1])
        self.rest.copy_(raw[2:])

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        first = torch.cat((self.first, self.first.new_zeros(1)))
        second = torch.cat((self.second, self.second.new_zeros(1)))
        if torch.is_grad_enabled():
            first.retain_grad()
            second.retain_grad()
            self.full_first = first
            self.full_second = second
        weight = torch.cat(
            (first.unsqueeze(0), second.unsqueeze(0), self.rest), dim=0
        )
        return F.embedding(idx, weight)
>>>>>>> REPLACE

<<<<<<< SEARCH
class FourFixedScaleLayerNorm(nn.Module):
    """Bias-free LayerNorm with four scales absorbed by following columns."""

    def __init__(self, normalized_shape: int, eps: float = 1e-5):
        super().__init__()
        self.normalized_shape = normalized_shape
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(normalized_shape - 4))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = torch.cat((self.weight, self.weight.new_ones(4)))
        return F.layer_norm(
            x, (self.normalized_shape,), weight, None, self.eps
        )
=======
class FiveFixedScaleLayerNorm(nn.Module):
    """Bias-free LayerNorm with five scales absorbed by following columns."""

    def __init__(self, normalized_shape: int, eps: float = 1e-5):
        super().__init__()
        self.normalized_shape = normalized_shape
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(normalized_shape - 5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = torch.cat((self.weight, self.weight.new_ones(5)))
        return F.layer_norm(
            x, (self.normalized_shape,), weight, None, self.eps
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.ln2 = FourFixedScaleLayerNorm(cfg.d_model)
=======
        self.ln2 = FiveFixedScaleLayerNorm(cfg.d_model)
>>>>>>> REPLACE

<<<<<<< SEARCH
    # Preserve full eight-coordinate AdamW dynamics for the positional,
    # attention-weight, attention-bias, terminal-bias, and four
    # terminal-weight gauges.
    gauge_params = [model.pos_emb.first]
=======
    # Preserve full eight-coordinate AdamW dynamics for both positional,
    # attention-weight, attention-bias, terminal-bias, and four
    # terminal-weight gauges.
    gauge_params = [model.pos_emb.first, model.pos_emb.second]
>>>>>>> REPLACE

<<<<<<< SEARCH
    # The last four ln2 scales are redundant with the corresponding fc1
    # columns. Keep their factorization only as optimizer-coordinate state,
    # while the model stores and uses the deduplicated effective columns.
    absorbed_scales = [
        torch.ones(4, device=device, dtype=blk.mlp.fc1.weight.dtype)
        for blk in model.blocks
    ]
    absorbed_weight_m = [
        torch.zeros_like(blk.mlp.fc1.weight[:, -4:])
        for blk in model.blocks
    ]
=======
    # The last five ln2 scales are redundant with the corresponding fc1
    # columns. Keep their factorization only as optimizer-coordinate state,
    # while the model stores and uses the deduplicated effective columns.
    absorbed_scales = [
        torch.ones(5, device=device, dtype=blk.mlp.fc1.weight.dtype)
        for blk in model.blocks
    ]
    absorbed_weight_m = [
        torch.zeros_like(blk.mlp.fc1.weight[:, -5:])
        for blk in model.blocks
    ]
>>>>>>> REPLACE

<<<<<<< SEARCH
        full_gauge_grads = [model.pos_emb.full_first.grad.detach()]
=======
        full_gauge_grads = [
            model.pos_emb.full_first.grad.detach(),
            model.pos_emb.full_second.grad.detach(),
        ]
>>>>>>> REPLACE

<<<<<<< SEARCH
            effective_grad = (
                blk.mlp.fc1.weight.grad[:, -4:].detach().clone()
            )
            virtual_weight = (
                blk.mlp.fc1.weight[:, -4:].detach()
                / virtual_scale.unsqueeze(0)
            )
=======
            effective_grad = (
                blk.mlp.fc1.weight.grad[:, -5:].detach().clone()
            )
            virtual_weight = (
                blk.mlp.fc1.weight[:, -5:].detach()
                / virtual_scale.unsqueeze(0)
            )
>>>>>>> REPLACE

<<<<<<< SEARCH
        for blk in model.blocks:
            blk.mlp.fc1.weight.grad[:, -4:].zero_()
=======
        for blk in model.blocks:
            blk.mlp.fc1.weight.grad[:, -5:].zero_()
>>>>>>> REPLACE

<<<<<<< SEARCH
                blk.mlp.fc1.weight[:, -4:].copy_(
                    virtual_weight * virtual_scale.unsqueeze(0)
                )
=======
                blk.mlp.fc1.weight[:, -5:].copy_(
                    virtual_weight * virtual_scale.unsqueeze(0)
                )
>>>>>>> REPLACE