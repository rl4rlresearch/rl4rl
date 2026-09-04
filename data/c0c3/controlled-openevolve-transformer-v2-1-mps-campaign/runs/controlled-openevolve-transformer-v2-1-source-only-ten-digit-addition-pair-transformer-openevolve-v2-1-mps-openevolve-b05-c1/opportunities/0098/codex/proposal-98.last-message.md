MECHANISM: Third orthonormal attention-null reuse for final normalization scale

HYPOTHESIS: Reusing projection column 3’s normalized row-common component as the remaining learned final-LayerNorm scale will reduce the model from 1,266 to 1,265 parameters while retaining at least 99% accuracy.

INTENDED_EDIT: Remove the final LayerNorm’s dedicated scale parameter, derive coordinate 0’s scale from the normalized mean of attention projection column 3, and initialize that null component to produce scale one.

EVIDENCE: Orthonormal null-direction reuse retained 100% accuracy for both a final-LayerNorm scale at 1,267 parameters and its bias at 1,266, directly supporting the same function-preserving reuse for the sole remaining dedicated scale.

<<<<<<< SEARCH
class FixedBiasLayerNorm(nn.Module):
    """LayerNorm with scales and bias stored in attention null directions."""

    def __init__(self, d_model: int):
        super().__init__()
        self.normalized_shape = (d_model,)
        self.weight = nn.Parameter(torch.ones(d_model - 7))

    def forward(
        self,
        x: torch.Tensor,
        shared_scales: torch.Tensor,
        shared_bias: torch.Tensor,
    ) -> torch.Tensor:
        weight = torch.cat(
            (
                self.weight[:1],
                self.weight.new_ones(1),
                shared_scales[1:2],
                self.weight.new_ones(2),
                shared_scales[:1],
                self.weight.new_ones(2),
            )
        )
=======
class FixedBiasLayerNorm(nn.Module):
    """LayerNorm with scales and bias stored in attention null directions."""

    def __init__(self, d_model: int):
        super().__init__()
        self.normalized_shape = (d_model,)

    def forward(
        self,
        x: torch.Tensor,
        shared_scales: torch.Tensor,
        shared_bias: torch.Tensor,
    ) -> torch.Tensor:
        weight = torch.cat(
            (
                shared_scales[2:3],
                shared_scales.new_ones(1),
                shared_scales[1:2],
                shared_scales.new_ones(2),
                shared_scales[:1],
                shared_scales.new_ones(2),
            )
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
            normalized_shared_bias_column = self.blocks[-1].attn.proj.weight[:, 2]
            normalized_shared_bias_column.sub_(
                normalized_shared_bias_column.mean()
            )
=======
            normalized_shared_bias_column = self.blocks[-1].attn.proj.weight[:, 2]
            normalized_shared_bias_column.sub_(
                normalized_shared_bias_column.mean()
            )
            normalized_shared_scale_column = self.blocks[-1].attn.proj.weight[:, 3]
            normalized_shared_scale_column.add_(
                target_mean - normalized_shared_scale_column.mean()
            )
>>>>>>> REPLACE

<<<<<<< SEARCH
        shared_final_scales = torch.stack(
            (
                final_proj_weight[:, 0].mean(),
                math.sqrt(self.cfg.d_model) * final_proj_weight[:, 1].mean(),
            )
        )
=======
        shared_final_scales = torch.stack(
            (
                final_proj_weight[:, 0].mean(),
                math.sqrt(self.cfg.d_model) * final_proj_weight[:, 1].mean(),
                math.sqrt(self.cfg.d_model) * final_proj_weight[:, 3].mean(),
            )
        )
>>>>>>> REPLACE