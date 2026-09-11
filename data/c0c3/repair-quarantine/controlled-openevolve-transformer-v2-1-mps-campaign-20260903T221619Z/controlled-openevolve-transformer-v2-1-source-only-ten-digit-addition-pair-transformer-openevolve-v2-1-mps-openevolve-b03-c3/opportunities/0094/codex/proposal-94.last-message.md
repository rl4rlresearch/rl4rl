MECHANISM: Fourth pre-attention LayerNorm scale absorption

HYPOTHESIS: A 1,377-parameter model will retain at least 99% accuracy because the qualified 1,378-parameter design reached 99.39%, and absorbing a fourth `ln1` scale extends the same exact q/k/v factorization that succeeded for three scales.

INTENDED_EDIT: Reproduce the qualified four-column attention-output gauge, then absorb four rather than three `ln1` scales while updating the ambient-gradient clipping and AdamW factorization slices accordingly.

EVIDENCE: Reference Design 3 achieved 99.39% accuracy at 1,378 parameters after three-scale absorption; unlike the failed fifth attention gauge and `fc1` row gauge, this applies the previously successful scale-absorption mechanism by one additional coordinate.

<<<<<<< SEARCH
class GaugeFixedAttentionProjection(nn.Module):
    """Linear projection with bias and three weight-column output gauges removed."""

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight_prefix = nn.ParameterList(
            [
                nn.Parameter(torch.empty(out_features - 1))
                for _ in range(3)
            ]
        )
        self.weight_rest = nn.Parameter(
            torch.empty(out_features, in_features - 3)
        )
=======
class GaugeFixedAttentionProjection(nn.Module):
    """Linear projection with bias and four weight-column output gauges removed."""

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight_prefix = nn.ParameterList(
            [
                nn.Parameter(torch.empty(out_features - 1))
                for _ in range(4)
            ]
        )
        self.weight_rest = nn.Parameter(
            torch.empty(out_features, in_features - 4)
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.weight_rest.copy_(raw_weight[:, 3:])
        self.bias.copy_(raw_bias[:-1] - raw_bias[-1])
=======
        self.weight_rest.copy_(raw_weight[:, 4:])
        self.bias.copy_(raw_bias[:-1] - raw_bias[-1])
>>>>>>> REPLACE

<<<<<<< SEARCH
class TwoAbsorbedScaleLayerNorm(nn.Module):
    """Bias-free LayerNorm with its final two scales absorbed downstream."""

    def __init__(self, normalized_shape: int, eps: float = 1e-5):
        super().__init__()
        self.normalized_shape = normalized_shape
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(normalized_shape - 2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = torch.cat((self.weight, self.weight.new_ones(2)))
=======
class FourAbsorbedScaleLayerNorm(nn.Module):
    """Bias-free LayerNorm with its final four scales absorbed downstream."""

    def __init__(self, normalized_shape: int, eps: float = 1e-5):
        super().__init__()
        self.normalized_shape = normalized_shape
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(normalized_shape - 4))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = torch.cat((self.weight, self.weight.new_ones(4)))
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.ln1 = TwoAbsorbedScaleLayerNorm(cfg.d_model)
=======
        self.ln1 = FourAbsorbedScaleLayerNorm(cfg.d_model)
>>>>>>> REPLACE

<<<<<<< SEARCH
                module.weight_rest.copy_(raw_weight[:, 3:])
                nn.init.zeros_(module.bias)
=======
                module.weight_rest.copy_(raw_weight[:, 4:])
                nn.init.zeros_(module.bias)
>>>>>>> REPLACE

<<<<<<< SEARCH
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
=======
    # The final four ln1 scales live only in optimizer-coordinate state;
    # q, k, and v store their products with the corresponding input columns.
    attention_scales = [
        torch.ones(
            4, device=device, dtype=blk.attn.q_proj.weight.dtype
        )
        for blk in model.blocks
    ]
    attention_weight_m = [
        torch.zeros_like(
            torch.cat(
                (
                    blk.attn.q_proj.weight[:, -4:],
                    blk.attn.k_proj.weight[:, -4:],
                    blk.attn.v_proj.weight[:, -4:],
                ),
                dim=0,
            )
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
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
=======
            effective_grad = torch.cat(
                (
                    blk.attn.q_proj.weight.grad[:, -4:],
                    blk.attn.k_proj.weight.grad[:, -4:],
                    blk.attn.v_proj.weight.grad[:, -4:],
                ),
                dim=0,
            ).detach().clone()
            effective_weight = torch.cat(
                (
                    blk.attn.q_proj.weight[:, -4:],
                    blk.attn.k_proj.weight[:, -4:],
                    blk.attn.v_proj.weight[:, -4:],
                ),
                dim=0,
            ).detach()
>>>>>>> REPLACE

<<<<<<< SEARCH
        for blk in model.blocks:
            blk.attn.q_proj.weight.grad[:, -2:].zero_()
            blk.attn.k_proj.weight.grad[:, -2:].zero_()
            blk.attn.v_proj.weight.grad[:, -2:].zero_()
            blk.mlp.fc1.weight.grad[:, -8:].zero_()
=======
        for blk in model.blocks:
            blk.attn.q_proj.weight.grad[:, -4:].zero_()
            blk.attn.k_proj.weight.grad[:, -4:].zero_()
            blk.attn.v_proj.weight.grad[:, -4:].zero_()
            blk.mlp.fc1.weight.grad[:, -8:].zero_()
>>>>>>> REPLACE

<<<<<<< SEARCH
                blk.attn.q_proj.weight[:, -2:].copy_(
                    effective_weight[:q_end]
                )
                blk.attn.k_proj.weight[:, -2:].copy_(
                    effective_weight[q_end:k_end]
                )
                blk.attn.v_proj.weight[:, -2:].copy_(
                    effective_weight[k_end:]
                )
=======
                blk.attn.q_proj.weight[:, -4:].copy_(
                    effective_weight[:q_end]
                )
                blk.attn.k_proj.weight[:, -4:].copy_(
                    effective_weight[q_end:k_end]
                )
                blk.attn.v_proj.weight[:, -4:].copy_(
                    effective_weight[k_end:]
                )
>>>>>>> REPLACE