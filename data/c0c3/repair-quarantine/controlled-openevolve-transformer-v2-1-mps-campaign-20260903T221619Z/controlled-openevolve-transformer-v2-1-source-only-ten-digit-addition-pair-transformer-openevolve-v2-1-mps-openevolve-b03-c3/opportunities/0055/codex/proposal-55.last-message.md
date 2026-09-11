MECHANISM: Ambient AdamW factorization of one `ln1` scale atop five absorbed `ln2` scales

HYPOTHESIS: The 1,534-parameter model will retain at least 99% accuracy because the removed `ln1` scale is exactly absorbable into the corresponding query, key, and value columns, while ambient-coordinate AdamW preserves the optimizer dynamics whose absence caused the naive `ln1` reduction to fail.

INTENDED_EDIT: Use five-scale `ln2` absorption from the qualified 1,535-parameter design, additionally absorb one `ln1` scale across all three attention input projections, and train both factorizations with ambient moments, clipping, and product materialization.

EVIDENCE: Five-scale `ln2` factorization achieved 99.95% at 1,535 parameters; more importantly, naive two-scale `ln2` fixing failed at 80.28% before ambient factorization restored 99.94%, directly motivating the same optimizer-side remedy for the naive one-scale `ln1` result of 71.65%.

<<<<<<< SEARCH
class OneFixedScaleLayerNorm(nn.Module):
    """Bias-free LayerNorm with one scale absorbed by the following linear."""

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


class MLP(nn.Module):
=======
class OneFixedScaleLayerNorm(nn.Module):
    """Bias-free LayerNorm with one scale absorbed by following projections."""

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


class MLP(nn.Module):
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.ln1 = nn.LayerNorm(cfg.d_model, bias=False)
        self.attn = CausalSelfAttention(cfg.d_model, cfg.n_head, cfg.dropout, cfg.max_seq_len)
        self.ln2 = OneFixedScaleLayerNorm(cfg.d_model)
=======
        self.ln1 = OneFixedScaleLayerNorm(cfg.d_model)
        self.attn = CausalSelfAttention(cfg.d_model, cfg.n_head, cfg.dropout, cfg.max_seq_len)
        self.ln2 = FiveFixedScaleLayerNorm(cfg.d_model)
>>>>>>> REPLACE

<<<<<<< SEARCH
    gauge_v = [torch.zeros_like(moment) for moment in gauge_m]
    gauge_step = 0

    sampler = TrainBatchSampler(train_cfg.batch_size, train_cfg.seed + 1337, reserved_hashes)
=======
    gauge_v = [torch.zeros_like(moment) for moment in gauge_m]
    gauge_step = 0

    # The final ln1 scale is redundant with the corresponding columns of all
    # three attention input projections. Retain its full factorization only as
    # optimizer state while the model stores the three effective columns.
    ln1_absorbed_scales = [
        torch.ones(1, device=device, dtype=blk.attn.q_proj.weight.dtype)
        for blk in model.blocks
    ]
    ln1_absorbed_weight_m = [
        torch.zeros(
            blk.attn.q_proj.out_features
            + blk.attn.k_proj.out_features
            + blk.attn.v_proj.out_features,
            device=device,
            dtype=blk.attn.q_proj.weight.dtype,
        )
        for blk in model.blocks
    ]
    ln1_absorbed_weight_v = [
        torch.zeros_like(moment) for moment in ln1_absorbed_weight_m
    ]
    ln1_absorbed_scale_m = [
        torch.zeros_like(scale) for scale in ln1_absorbed_scales
    ]
    ln1_absorbed_scale_v = [
        torch.zeros_like(moment) for moment in ln1_absorbed_scale_m
    ]
    ln1_absorbed_step = 0

    # Likewise, retain the last five ln2 scales and their fc1 columns only as
    # ambient optimizer coordinates; the learned model stores their products.
    ln2_absorbed_scales = [
        torch.ones(5, device=device, dtype=blk.mlp.fc1.weight.dtype)
        for blk in model.blocks
    ]
    ln2_absorbed_weight_m = [
        torch.zeros_like(blk.mlp.fc1.weight[:, -5:])
        for blk in model.blocks
    ]
    ln2_absorbed_weight_v = [
        torch.zeros_like(moment) for moment in ln2_absorbed_weight_m
    ]
    ln2_absorbed_scale_m = [
        torch.zeros_like(scale) for scale in ln2_absorbed_scales
    ]
    ln2_absorbed_scale_v = [
        torch.zeros_like(moment) for moment in ln2_absorbed_scale_m
    ]
    ln2_absorbed_step = 0

    sampler = TrainBatchSampler(train_cfg.batch_size, train_cfg.seed + 1337, reserved_hashes)
>>>>>>> REPLACE

<<<<<<< SEARCH
        clip_scale = 1.0
        if train_cfg.grad_clip > 0:
            # Stored gradients contain the first seven coordinates; include
            # every omitted ambient coordinate in the clipping norm.
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

        gauge_step += 1
=======
        ln1_absorbed_grads = []
        for blk, virtual_scale in zip(
            model.blocks, ln1_absorbed_scales
        ):
            effective_weight = torch.cat(
                (
                    blk.attn.q_proj.weight[:, -1],
                    blk.attn.k_proj.weight[:, -1],
                    blk.attn.v_proj.weight[:, -1],
                )
            ).detach()
            effective_grad = torch.cat(
                (
                    blk.attn.q_proj.weight.grad[:, -1],
                    blk.attn.k_proj.weight.grad[:, -1],
                    blk.attn.v_proj.weight.grad[:, -1],
                )
            ).detach().clone()
            virtual_weight = effective_weight / virtual_scale[0]
            ambient_weight_grad = effective_grad * virtual_scale[0]
            ambient_scale_grad = (
                effective_grad * virtual_weight
            ).sum().reshape(1)
            ln1_absorbed_grads.append(
                (
                    effective_grad,
                    virtual_weight,
                    ambient_weight_grad,
                    ambient_scale_grad,
                )
            )

        ln2_absorbed_grads = []
        for blk, virtual_scale in zip(
            model.blocks, ln2_absorbed_scales
        ):
            effective_grad = (
                blk.mlp.fc1.weight.grad[:, -5:].detach().clone()
            )
            virtual_weight = (
                blk.mlp.fc1.weight[:, -5:].detach()
                / virtual_scale.unsqueeze(0)
            )
            ambient_weight_grad = (
                effective_grad * virtual_scale.unsqueeze(0)
            )
            ambient_scale_grad = (
                effective_grad * virtual_weight
            ).sum(dim=0)
            ln2_absorbed_grads.append(
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
                grad_sq = grad_sq + full_grad[-1].float().square()
            for (
                effective_grad,
                _,
                ambient_weight_grad,
                ambient_scale_grad,
            ) in ln1_absorbed_grads:
                grad_sq = (
                    grad_sq
                    - effective_grad.float().square().sum()
                    + ambient_weight_grad.float().square().sum()
                    + ambient_scale_grad.float().square().sum()
                )
            for (
                effective_grad,
                _,
                ambient_weight_grad,
                ambient_scale_grad,
            ) in ln2_absorbed_grads:
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

        # The ordinary optimizer updates neither set of effective columns.
        for blk in model.blocks:
            blk.attn.q_proj.weight.grad[:, -1].zero_()
            blk.attn.k_proj.weight.grad[:, -1].zero_()
            blk.attn.v_proj.weight.grad[:, -1].zero_()
            blk.mlp.fc1.weight.grad[:, -5:].zero_()

        optimizer.step()

        ln1_absorbed_step += 1
        ln1_bc1 = 1.0 - 0.9 ** ln1_absorbed_step
        ln1_bc2 = 1.0 - 0.999 ** ln1_absorbed_step
        for i, (blk, absorbed_grad) in enumerate(
            zip(model.blocks, ln1_absorbed_grads)
        ):
            (
                _,
                virtual_weight,
                ambient_weight_grad,
                ambient_scale_grad,
            ) = absorbed_grad
            weight_grad = ambient_weight_grad * clip_scale
            scale_grad = ambient_scale_grad * clip_scale
            weight_moment = ln1_absorbed_weight_m[i]
            weight_variance = ln1_absorbed_weight_v[i]
            scale_moment = ln1_absorbed_scale_m[i]
            scale_variance = ln1_absorbed_scale_v[i]
            virtual_scale = ln1_absorbed_scales[i]

            weight_moment.mul_(0.9).add_(weight_grad, alpha=0.1)
            weight_variance.mul_(0.999).addcmul_(
                weight_grad, weight_grad, value=0.001
            )
            scale_moment.mul_(0.9).add_(scale_grad, alpha=0.1)
            scale_variance.mul_(0.999).addcmul_(
                scale_grad, scale_grad, value=0.001
            )
            weight_direction = (
                weight_moment / ln1_bc1
            ) / (
                (weight_variance / ln1_bc2).sqrt() + 1e-8
            )
            scale_direction = (
                scale_moment / ln1_bc1
            ) / (
                (scale_variance / ln1_bc2).sqrt() + 1e-8
            )

            with torch.no_grad():
                decay = 1.0 - lr_now * train_cfg.weight_decay
                virtual_weight.mul_(decay).add_(
                    weight_direction, alpha=-lr_now
                )
                virtual_scale.mul_(decay).add_(
                    scale_direction, alpha=-lr_now
                )
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

        ln2_absorbed_step += 1
        ln2_bc1 = 1.0 - 0.9 ** ln2_absorbed_step
        ln2_bc2 = 1.0 - 0.999 ** ln2_absorbed_step
        for i, (blk, absorbed_grad) in enumerate(
            zip(model.blocks, ln2_absorbed_grads)
        ):
            (
                _,
                virtual_weight,
                ambient_weight_grad,
                ambient_scale_grad,
            ) = absorbed_grad
            weight_grad = ambient_weight_grad * clip_scale
            scale_grad = ambient_scale_grad * clip_scale
            weight_moment = ln2_absorbed_weight_m[i]
            weight_variance = ln2_absorbed_weight_v[i]
            scale_moment = ln2_absorbed_scale_m[i]
            scale_variance = ln2_absorbed_scale_v[i]
            virtual_scale = ln2_absorbed_scales[i]

            weight_moment.mul_(0.9).add_(weight_grad, alpha=0.1)
            weight_variance.mul_(0.999).addcmul_(
                weight_grad, weight_grad, value=0.001
            )
            scale_moment.mul_(0.9).add_(scale_grad, alpha=0.1)
            scale_variance.mul_(0.999).addcmul_(
                scale_grad, scale_grad, value=0.001
            )
            weight_direction = (
                weight_moment / ln2_bc1
            ) / (
                (weight_variance / ln2_bc2).sqrt() + 1e-8
            )
            scale_direction = (
                scale_moment / ln2_bc1
            ) / (
                (scale_variance / ln2_bc2).sqrt() + 1e-8
            )

            with torch.no_grad():
                decay = 1.0 - lr_now * train_cfg.weight_decay
                virtual_weight.mul_(decay).add_(
                    weight_direction, alpha=-lr_now
                )
                virtual_scale.mul_(decay).add_(
                    scale_direction, alpha=-lr_now
                )
                blk.mlp.fc1.weight[:, -5:].copy_(
                    virtual_weight * virtual_scale.unsqueeze(0)
                )

        gauge_step += 1
>>>>>>> REPLACE