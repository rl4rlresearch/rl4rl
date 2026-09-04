MECHANISM: Single ln1 scale absorption into query, key, and value projection columns

HYPOTHESIS: A 1,526-parameter model will retain at least 99% accuracy because the verified 1,527-parameter design successfully absorbs all eight ln2 scales, while one ln1 scale is likewise exactly absorbable into the corresponding q, k, and v input columns with ambient AdamW dynamics preserved.

INTENDED_EDIT: Replace the final learned ln1 scale with a fixed unit scale, store its products in the final q/k/v columns, and optimize the omitted scale and virtual projection columns through ambient-coordinate clipping, moments, decay, and updates.

EVIDENCE: The current complete ln2 scale-absorption design achieved 99.94% at 1,527 parameters; applying one instance of the same exact factorization to ln1 is a controlled new reduction axis after fourth-attention and seventh-terminal gauges fell below 99%.

<<<<<<< SEARCH
class FullyAbsorbedScaleLayerNorm(nn.Module):
    """Parameter-free LayerNorm whose scales live in following fc1 columns."""
=======
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


class FullyAbsorbedScaleLayerNorm(nn.Module):
    """Parameter-free LayerNorm whose scales live in following fc1 columns."""
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.ln1 = nn.LayerNorm(cfg.d_model, bias=False)
=======
        self.ln1 = OneAbsorbedScaleLayerNorm(cfg.d_model)
>>>>>>> REPLACE

<<<<<<< SEARCH
    gauge_step = 0

    # All eight ln2 scales are represented only as optimizer-coordinate
=======
    gauge_step = 0

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

    # All eight ln2 scales are represented only as optimizer-coordinate
>>>>>>> REPLACE

<<<<<<< SEARCH
        absorbed_grads = []
=======
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

        absorbed_grads = []
>>>>>>> REPLACE

<<<<<<< SEARCH
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
=======
            for (
                effective_grad,
                _,
                ambient_weight_grad,
                ambient_scale_grad,
            ) in attention_absorbed_grads:
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
            ) in absorbed_grads:
                grad_sq = (
                    grad_sq
                    - effective_grad.float().square().sum()
                    + ambient_weight_grad.float().square().sum()
                    + ambient_scale_grad.float().square().sum()
                )
            total_norm = float(grad_sq.sqrt().item())
>>>>>>> REPLACE

<<<<<<< SEARCH
        for blk in model.blocks:
            blk.mlp.fc1.weight.grad[:, -8:].zero_()

        optimizer.step()
=======
        for blk in model.blocks:
            blk.attn.q_proj.weight.grad[:, -1].zero_()
            blk.attn.k_proj.weight.grad[:, -1].zero_()
            blk.attn.v_proj.weight.grad[:, -1].zero_()
            blk.mlp.fc1.weight.grad[:, -8:].zero_()

        optimizer.step()
>>>>>>> REPLACE

<<<<<<< SEARCH
        optimizer.step()

        absorbed_step += 1
=======
        optimizer.step()

        attention_step += 1
        attention_bc1 = 1.0 - 0.9 ** attention_step
        attention_bc2 = 1.0 - 0.999 ** attention_step
        for i, (blk, attention_grad) in enumerate(
            zip(model.blocks, attention_absorbed_grads)
        ):
            (
                _,
                virtual_weight,
                ambient_weight_grad,
                ambient_scale_grad,
            ) = attention_grad
            weight_grad = ambient_weight_grad * clip_scale
            scale_grad = ambient_scale_grad * clip_scale
            weight_moment = attention_weight_m[i]
            weight_variance = attention_weight_v[i]
            scale_moment = attention_scale_m[i]
            scale_variance = attention_scale_v[i]
            virtual_scale = attention_scales[i]

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
                weight_moment / attention_bc1
            ) / (
                (weight_variance / attention_bc2).sqrt() + 1e-8
            )
            scale_direction = (
                scale_moment / attention_bc1
            ) / (
                (scale_variance / attention_bc2).sqrt() + 1e-8
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

        absorbed_step += 1
>>>>>>> REPLACE