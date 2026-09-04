MECHANISM: Third positional-row scalar-shift gauge with ambient AdamW

HYPOTHESIS: A 1,523-parameter model will retain at least 99% accuracy because the verified 1,524-parameter design achieved 99.92%, while a third positional row has the same exact pre-LayerNorm scalar-shift symmetry as the two already removed.

INTENDED_EDIT: Reproduce the qualified two-scale ln1 absorption and two positional gauges, then gauge-fix a third positional row while preserving all omitted coordinates in initialization, gradient clipping, AdamW moments, decay, and updates.

EVIDENCE: Reference Design 3 reached 99.92% with two absorbed ln1 scales and two positional scalar gauges; extending the successful positional symmetry is the smallest controlled reduction after a third ln1 absorption fell to 98.85%.

<<<<<<< SEARCH
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
=======
class GaugeFixedPositionEmbedding(nn.Module):
    """Embedding with three independent positional scalar shifts removed."""

    def __init__(self, num_embeddings: int, embedding_dim: int):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.first = nn.Parameter(torch.empty(embedding_dim - 1))
        self.second = nn.Parameter(torch.empty(embedding_dim - 1))
        self.third = nn.Parameter(torch.empty(embedding_dim - 1))
        self.rest = nn.Parameter(
            torch.empty(num_embeddings - 3, embedding_dim)
        )
        self.full_first = None
        self.full_second = None
        self.full_third = None
        self.reset_parameters()

    @torch.no_grad()
    def reset_parameters(self, std: float = 1.0) -> None:
        raw = self.first.new_empty(
            self.num_embeddings, self.embedding_dim
        )
        nn.init.normal_(raw, mean=0.0, std=std)
        self.first.copy_(raw[0, :-1] - raw[0, -1])
        self.second.copy_(raw[1, :-1] - raw[1, -1])
        self.third.copy_(raw[2, :-1] - raw[2, -1])
        self.rest.copy_(raw[3:])

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        first = torch.cat((self.first, self.first.new_zeros(1)))
        second = torch.cat((self.second, self.second.new_zeros(1)))
        third = torch.cat((self.third, self.third.new_zeros(1)))
        if torch.is_grad_enabled():
            first.retain_grad()
            second.retain_grad()
            third.retain_grad()
            self.full_first = first
            self.full_second = second
            self.full_third = third
        weight = torch.cat(
            (
                first.unsqueeze(0),
                second.unsqueeze(0),
                third.unsqueeze(0),
                self.rest,
            ),
            dim=0,
        )
        return F.embedding(idx, weight)
>>>>>>> REPLACE

<<<<<<< SEARCH
class OneAbsorbedScaleLayerNorm(nn.Module):
    """Bias-free LayerNorm with its final scale absorbed downstream."""

    def __init__(self, normalized_shape: int, eps: float = 1e-5):
        super().__init__()
        self.normalized_shape = normalized_shape
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(normalized_shape - 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = torch.cat((self.weight, self.weight.new_ones(1)))
        return F.layer_norm(
            x, (self.normalized_shape,), weight, None, self.eps
        )
=======
class TwoAbsorbedScaleLayerNorm(nn.Module):
    """Bias-free LayerNorm with its final two scales absorbed downstream."""

    def __init__(self, normalized_shape: int, eps: float = 1e-5):
        super().__init__()
        self.normalized_shape = normalized_shape
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(normalized_shape - 2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = torch.cat((self.weight, self.weight.new_ones(2)))
        return F.layer_norm(
            x, (self.normalized_shape,), weight, None, self.eps
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.ln1 = OneAbsorbedScaleLayerNorm(cfg.d_model)
=======
        self.ln1 = TwoAbsorbedScaleLayerNorm(cfg.d_model)
>>>>>>> REPLACE

<<<<<<< SEARCH
    # Optimize omitted gauge coordinates in their full ambient spaces.
    gauge_params = [model.token_emb.weight, model.pos_emb.first]
=======
    # Optimize omitted gauge coordinates in their full ambient spaces.
    gauge_params = [
        model.token_emb.weight,
        model.pos_emb.first,
        model.pos_emb.second,
        model.pos_emb.third,
    ]
>>>>>>> REPLACE

<<<<<<< SEARCH
    # The final ln1 scale is represented only as optimizer-coordinate state;
    # the q, k, and v projections store its products with their final columns.
    attention_scales = [
        torch.ones(
            1, device=device, dtype=blk.attn.q_proj.weight.dtype
        )
        for blk in model.blocks
    ]
    attention_weight_m = [
        torch.zeros_like(
            torch.cat(
                (
                    blk.attn.q_proj.weight[:, -1],
                    blk.attn.k_proj.weight[:, -1],
                    blk.attn.v_proj.weight[:, -1],
                )
            )
        )
        for blk in model.blocks
    ]
    attention_weight_v = [
        torch.zeros_like(moment) for moment in attention_weight_m
    ]
    attention_scale_m = [
        torch.zeros_like(scale) for scale in attention_scales
    ]
    attention_scale_v = [
        torch.zeros_like(moment) for moment in attention_scale_m
    ]
    attention_step = 0
=======
    # The final two ln1 scales are represented only as optimizer-coordinate
    # state; q, k, and v store their products with the corresponding columns.
    attention_scales = [
        torch.ones(
            2, device=device, dtype=blk.attn.q_proj.weight.dtype
        )
        for blk in model.blocks
    ]
    attention_weight_m = [
        torch.zeros_like(
            torch.cat(
                (
                    blk.attn.q_proj.weight[:, -2:],
                    blk.attn.k_proj.weight[:, -2:],
                    blk.attn.v_proj.weight[:, -2:],
                ),
                dim=0,
            )
        )
        for blk in model.blocks
    ]
    attention_weight_v = [
        torch.zeros_like(moment) for moment in attention_weight_m
    ]
    attention_scale_m = [
        torch.zeros_like(scale) for scale in attention_scales
    ]
    attention_scale_v = [
        torch.zeros_like(moment) for moment in attention_scale_m
    ]
    attention_step = 0
>>>>>>> REPLACE

<<<<<<< SEARCH
        full_gauge_grads = [
            model.token_emb.full_weight.grad.detach().reshape(-1),
            model.pos_emb.full_first.grad.detach(),
        ]
=======
        full_gauge_grads = [
            model.token_emb.full_weight.grad.detach().reshape(-1),
            model.pos_emb.full_first.grad.detach(),
            model.pos_emb.full_second.grad.detach(),
            model.pos_emb.full_third.grad.detach(),
        ]
>>>>>>> REPLACE

<<<<<<< SEARCH
        attention_absorbed_grads = []
        for blk, virtual_scale in zip(
            model.blocks, attention_scales
        ):
            effective_grad = torch.cat(
                (
                    blk.attn.q_proj.weight.grad[:, -1],
                    blk.attn.k_proj.weight.grad[:, -1],
                    blk.attn.v_proj.weight.grad[:, -1],
                )
            ).detach().clone()
            effective_weight = torch.cat(
                (
                    blk.attn.q_proj.weight[:, -1],
                    blk.attn.k_proj.weight[:, -1],
                    blk.attn.v_proj.weight[:, -1],
                )
            ).detach()
            virtual_weight = effective_weight / virtual_scale[0]
            ambient_weight_grad = effective_grad * virtual_scale[0]
            ambient_scale_grad = (
                effective_grad * virtual_weight
            ).sum().reshape(1)
            attention_absorbed_grads.append(
                (
                    effective_grad,
                    virtual_weight,
                    ambient_weight_grad,
                    ambient_scale_grad,
                )
            )
=======
        attention_absorbed_grads = []
        for blk, virtual_scale in zip(
            model.blocks, attention_scales
        ):
            effective_grad = torch.cat(
                (
                    blk.attn.q_proj.weight.grad[:, -2:],
                    blk.attn.k_proj.weight.grad[:, -2:],
                    blk.attn.v_proj.weight.grad[:, -2:],
                ),
                dim=0,
            ).detach().clone()
            effective_weight = torch.cat(
                (
                    blk.attn.q_proj.weight[:, -2:],
                    blk.attn.k_proj.weight[:, -2:],
                    blk.attn.v_proj.weight[:, -2:],
                ),
                dim=0,
            ).detach()
            virtual_weight = (
                effective_weight / virtual_scale.unsqueeze(0)
            )
            ambient_weight_grad = (
                effective_grad * virtual_scale.unsqueeze(0)
            )
            ambient_scale_grad = (
                effective_grad * virtual_weight
            ).sum(dim=0)
            attention_absorbed_grads.append(
                (
                    effective_grad,
                    virtual_weight,
                    ambient_weight_grad,
                    ambient_scale_grad,
                )
            )
>>>>>>> REPLACE

<<<<<<< SEARCH
        for blk in model.blocks:
            blk.attn.q_proj.weight.grad[:, -1].zero_()
            blk.attn.k_proj.weight.grad[:, -1].zero_()
            blk.attn.v_proj.weight.grad[:, -1].zero_()
            blk.mlp.fc1.weight.grad[:, -8:].zero_()
=======
        for blk in model.blocks:
            blk.attn.q_proj.weight.grad[:, -2:].zero_()
            blk.attn.k_proj.weight.grad[:, -2:].zero_()
            blk.attn.v_proj.weight.grad[:, -2:].zero_()
            blk.mlp.fc1.weight.grad[:, -8:].zero_()
>>>>>>> REPLACE

<<<<<<< SEARCH
                effective_weight = virtual_weight * virtual_scale[0]
                q_end = blk.attn.q_proj.out_features
                k_end = q_end + blk.attn.k_proj.out_features
                blk.attn.q_proj.weight[:, -1].copy_(
                    effective_weight[:q_end]
                )
                blk.attn.k_proj.weight[:, -1].copy_(
                    effective_weight[q_end:k_end]
                )
                blk.attn.v_proj.weight[:, -1].copy_(
                    effective_weight[k_end:]
                )
=======
                effective_weight = (
                    virtual_weight * virtual_scale.unsqueeze(0)
                )
                q_end = blk.attn.q_proj.out_features
                k_end = q_end + blk.attn.k_proj.out_features
                blk.attn.q_proj.weight[:, -2:].copy_(
                    effective_weight[:q_end]
                )
                blk.attn.k_proj.weight[:, -2:].copy_(
                    effective_weight[q_end:k_end]
                )
                blk.attn.v_proj.weight[:, -2:].copy_(
                    effective_weight[k_end:]
                )
>>>>>>> REPLACE