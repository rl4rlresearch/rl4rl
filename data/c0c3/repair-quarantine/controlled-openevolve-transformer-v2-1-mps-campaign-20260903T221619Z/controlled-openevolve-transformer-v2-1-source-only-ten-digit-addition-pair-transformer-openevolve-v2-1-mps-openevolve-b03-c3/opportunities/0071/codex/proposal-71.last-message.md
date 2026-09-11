MECHANISM: Second ln1 scale absorption into query, key, and value projection columns

HYPOTHESIS: A 1,525-parameter model will retain at least 99% accuracy because the qualified 1,526-parameter design reached 99.61%, while a second ln1 scale has the same exact q/k/v column-factorization symmetry and preserves full ambient AdamW dynamics.

INTENDED_EDIT: Absorb all eight ln2 scales into fc1 and the final two ln1 scales into the corresponding q, k, and v columns, retaining virtual scale/weight gradients, clipping, moments, decay, and updates.

EVIDENCE: The 1,526-parameter design achieved 99.61% after absorbing all ln2 scales and one ln1 scale; extending that proven ln1 factorization by one coordinate is the smallest controlled reduction below the best qualified parameter count.

<<<<<<< SEARCH
class SixFixedScaleLayerNorm(nn.Module):
    """Bias-free LayerNorm with six scales absorbed by following columns."""

    def __init__(self, normalized_shape: int, eps: float = 1e-5):
        super().__init__()
        self.normalized_shape = normalized_shape
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(normalized_shape - 6))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = torch.cat((self.weight, self.weight.new_ones(6)))
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
>>>>>>> REPLACE

<<<<<<< SEARCH
class Block(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.d_model, bias=False)
        self.attn = CausalSelfAttention(cfg.d_model, cfg.n_head, cfg.dropout, cfg.max_seq_len)
        self.ln2 = SixFixedScaleLayerNorm(cfg.d_model)
        self.mlp = MLP(cfg.d_model, cfg.d_ff, cfg.dropout)
=======
class Block(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.ln1 = TwoAbsorbedScaleLayerNorm(cfg.d_model)
        self.attn = CausalSelfAttention(cfg.d_model, cfg.n_head, cfg.dropout, cfg.max_seq_len)
        self.ln2 = FullyAbsorbedScaleLayerNorm(cfg.d_model)
        self.mlp = MLP(cfg.d_model, cfg.d_ff, cfg.dropout)
>>>>>>> REPLACE

<<<<<<< SEARCH
    # The last six ln2 scales are redundant with the corresponding fc1
    # columns. Keep their factorization only as optimizer-coordinate state,
    # while the model stores and uses the deduplicated effective columns.
    absorbed_scales = [
        torch.ones(6, device=device, dtype=blk.mlp.fc1.weight.dtype)
        for blk in model.blocks
    ]
    absorbed_weight_m = [
        torch.zeros_like(blk.mlp.fc1.weight[:, -6:])
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
=======
    # The final two ln1 scales live only in optimizer-coordinate state;
    # q, k, and v store their products with the corresponding input columns.
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

    # All eight ln2 scales live only in optimizer-coordinate state; fc1
    # stores and uses their products with the ambient weight columns.
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

        absorbed_grads = []
>>>>>>> REPLACE

<<<<<<< SEARCH
            effective_grad = (
                blk.mlp.fc1.weight.grad[:, -6:].detach().clone()
            )
            virtual_weight = (
                blk.mlp.fc1.weight[:, -6:].detach()
                / virtual_scale.unsqueeze(0)
            )
=======
            effective_grad = (
                blk.mlp.fc1.weight.grad[:, -8:].detach().clone()
            )
            virtual_weight = (
                blk.mlp.fc1.weight[:, -8:].detach()
                / virtual_scale.unsqueeze(0)
            )
>>>>>>> REPLACE

<<<<<<< SEARCH
            for (
                effective_grad,
                _,
                ambient_weight_grad,
                ambient_scale_grad,
            ) in absorbed_grads:
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
>>>>>>> REPLACE

<<<<<<< SEARCH
        # These effective columns are updated below through their ambient
        # weight/scale factorization rather than by the ordinary optimizer.
        for blk in model.blocks:
            blk.mlp.fc1.weight.grad[:, -6:].zero_()
=======
        # These effective columns are updated below through their ambient
        # weight/scale factorizations rather than by the ordinary optimizer.
        for blk in model.blocks:
            blk.attn.q_proj.weight.grad[:, -2:].zero_()
            blk.attn.k_proj.weight.grad[:, -2:].zero_()
            blk.attn.v_proj.weight.grad[:, -2:].zero_()
            blk.mlp.fc1.weight.grad[:, -8:].zero_()
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

            weight_moment.mul_(0.9).add_(weight_grad, alpha=0.1)
            weight_variance.mul_(0.999).addcmul_(
                weight_grad, weight_grad, value=0.001
            )
            scale_moment.mul_(0.9).add_(scale_grad, alpha=0.1)
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

        absorbed_step += 1
>>>>>>> REPLACE

<<<<<<< SEARCH
                blk.mlp.fc1.weight[:, -6:].copy_(
                    virtual_weight * virtual_scale.unsqueeze(0)
                )
=======
                blk.mlp.fc1.weight[:, -8:].copy_(
                    virtual_weight * virtual_scale.unsqueeze(0)
                )
>>>>>>> REPLACE