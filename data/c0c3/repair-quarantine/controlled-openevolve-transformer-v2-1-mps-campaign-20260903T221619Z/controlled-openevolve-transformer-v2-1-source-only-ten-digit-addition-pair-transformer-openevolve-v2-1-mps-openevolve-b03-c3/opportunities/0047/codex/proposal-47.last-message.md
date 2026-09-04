MECHANISM: Fourth multiplicatively absorbed LayerNorm scale with ambient AdamW

HYPOTHESIS: Extending the verified three-scale optimizer-side factorization to a fourth `ln2` scale will reduce the model from 1,537 to 1,536 parameters while retaining at least 99% accuracy, because the additional scale is exactly absorbable into its `fc1` column and its separate AdamW dynamics remain preserved outside the deduplicated model.

INTENDED_EDIT: Absorb four rather than three trailing `ln2` scales into the corresponding `fc1` columns, expanding the existing ambient-coordinate moments, clipping correction, and materialization update accordingly.

EVIDENCE: The three-scale ambient-factorization design achieved 99.97% accuracy at 1,537 parameters, improving on the already-qualified two-scale design; this makes one additional coordinate along the same successful exact-factorization axis the smallest controlled next reduction.

<<<<<<< SEARCH
class ThreeFixedScaleLayerNorm(nn.Module):
    """Bias-free LayerNorm with three scales absorbed by following columns."""

    def __init__(self, normalized_shape: int, eps: float = 1e-5):
        super().__init__()
        self.normalized_shape = normalized_shape
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(normalized_shape - 3))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = torch.cat((self.weight, self.weight.new_ones(3)))
=======
class FourFixedScaleLayerNorm(nn.Module):
    """Bias-free LayerNorm with four scales absorbed by following columns."""

    def __init__(self, normalized_shape: int, eps: float = 1e-5):
        super().__init__()
        self.normalized_shape = normalized_shape
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(normalized_shape - 4))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = torch.cat((self.weight, self.weight.new_ones(4)))
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.ln2 = ThreeFixedScaleLayerNorm(cfg.d_model)
=======
        self.ln2 = FourFixedScaleLayerNorm(cfg.d_model)
>>>>>>> REPLACE

<<<<<<< SEARCH
    # The last three ln2 scales are redundant with the corresponding fc1
    # columns. Keep their factorization only as optimizer-coordinate state,
    # while the model stores and uses the deduplicated effective columns.
    absorbed_scales = [
        torch.ones(3, device=device, dtype=blk.mlp.fc1.weight.dtype)
        for blk in model.blocks
    ]
    absorbed_weight_m = [
        torch.zeros_like(blk.mlp.fc1.weight[:, -3:])
        for blk in model.blocks
    ]
=======
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
>>>>>>> REPLACE

<<<<<<< SEARCH
            effective_grad = (
                blk.mlp.fc1.weight.grad[:, -3:].detach().clone()
            )
            virtual_weight = (
                blk.mlp.fc1.weight[:, -3:].detach()
                / virtual_scale.unsqueeze(0)
            )
=======
            effective_grad = (
                blk.mlp.fc1.weight.grad[:, -4:].detach().clone()
            )
            virtual_weight = (
                blk.mlp.fc1.weight[:, -4:].detach()
                / virtual_scale.unsqueeze(0)
            )
>>>>>>> REPLACE

<<<<<<< SEARCH
        for blk in model.blocks:
            blk.mlp.fc1.weight.grad[:, -3:].zero_()
=======
        for blk in model.blocks:
            blk.mlp.fc1.weight.grad[:, -4:].zero_()
>>>>>>> REPLACE

<<<<<<< SEARCH
                blk.mlp.fc1.weight[:, -3:].copy_(
                    virtual_weight * virtual_scale.unsqueeze(0)
                )
=======
                blk.mlp.fc1.weight[:, -4:].copy_(
                    virtual_weight * virtual_scale.unsqueeze(0)
                )
>>>>>>> REPLACE