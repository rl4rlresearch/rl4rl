MECHANISM: Fifth multiplicatively absorbed LayerNorm scale with ambient AdamW

HYPOTHESIS: A 1,535-parameter model will retain at least 99% accuracy because the fifth `ln2` scale is exactly absorbable into its `fc1` column and the verified ambient-coordinate AdamW factorization preserves its separate optimization dynamics.

INTENDED_EDIT: Expand the current two-scale absorption to five trailing `ln2` scales, including the virtual scales, optimizer moments, clipping correction, and effective-column materialization.

EVIDENCE: Four-scale ambient factorization achieved 100% accuracy at 1,536 parameters, after three scales achieved 99.97%; extending the same successful exact-factorization axis by one coordinate is the smallest informative reduction.

<<<<<<< SEARCH
class TwoFixedScaleLayerNorm(nn.Module):
    """Bias-free LayerNorm with two scales absorbed by following columns."""

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
        self.ln2 = TwoFixedScaleLayerNorm(cfg.d_model)
=======
        self.ln2 = FiveFixedScaleLayerNorm(cfg.d_model)
>>>>>>> REPLACE

<<<<<<< SEARCH
    # The last two ln2 scales are redundant with the corresponding fc1
    # columns. Keep their factorization only as optimizer-coordinate state,
    # while the model stores and uses the deduplicated effective columns.
    absorbed_scales = [
        torch.ones(2, device=device, dtype=blk.mlp.fc1.weight.dtype)
        for blk in model.blocks
    ]
    absorbed_weight_m = [
        torch.zeros_like(blk.mlp.fc1.weight[:, -2:])
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
            effective_grad = (
                blk.mlp.fc1.weight.grad[:, -2:].detach().clone()
            )
            virtual_weight = (
                blk.mlp.fc1.weight[:, -2:].detach()
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
            blk.mlp.fc1.weight.grad[:, -2:].zero_()
=======
        for blk in model.blocks:
            blk.mlp.fc1.weight.grad[:, -5:].zero_()
>>>>>>> REPLACE

<<<<<<< SEARCH
                blk.mlp.fc1.weight[:, -2:].copy_(
                    virtual_weight * virtual_scale.unsqueeze(0)
                )
=======
                blk.mlp.fc1.weight[:, -5:].copy_(
                    virtual_weight * virtual_scale.unsqueeze(0)
                )
>>>>>>> REPLACE