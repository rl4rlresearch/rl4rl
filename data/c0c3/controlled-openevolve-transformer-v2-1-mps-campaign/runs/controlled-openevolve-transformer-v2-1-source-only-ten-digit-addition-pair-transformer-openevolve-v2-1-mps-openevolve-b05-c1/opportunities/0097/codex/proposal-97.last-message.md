MECHANISM: Orthonormal attention-null reuse for final normalization bias

HYPOTHESIS: Replacing the dedicated final-LayerNorm bias with the normalized row-common component of attention projection column 2 will reduce the model from 1,267 to 1,266 parameters while retaining at least 99% accuracy, because it preserves the initialized function and gives the reused bias unit gradient norm.

INTENDED_EDIT: Remove the final-LayerNorm bias parameter, derive it as `sqrt(d_model)` times projection column 2’s mean, and initialize that null component to zero.

EVIDENCE: Raw-mean reuse for this bias reached only 74.13%, while changing the analogous second-scale reuse from a raw mean to an orthonormal mean produced 100% accuracy at 1,267 parameters; this directly tests whether conditioning caused the bias-reuse failure.

<<<<<<< SEARCH
class FixedBiasLayerNorm(nn.Module):
    """LayerNorm with two trainable scales stored in attention null directions."""

    def __init__(self, d_model: int):
        super().__init__()
        self.normalized_shape = (d_model,)
        self.weight = nn.Parameter(torch.ones(d_model - 7))
        self.bias = nn.Parameter(torch.zeros(d_model - 7))

    def forward(
        self, x: torch.Tensor, shared_scales: torch.Tensor
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
        return F.layer_norm(
            x,
            self.normalized_shape,
            weight,
            F.pad(self.bias, (0, 7)),
        )
=======
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
        return F.layer_norm(
            x,
            self.normalized_shape,
            weight,
            F.pad(shared_bias.reshape(1), (0, 7)),
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
            normalized_shared_column = self.blocks[-1].attn.proj.weight[:, 1]
            target_mean = 1.0 / math.sqrt(cfg.d_model)
            normalized_shared_column.add_(
                target_mean - normalized_shared_column.mean()
            )
=======
            normalized_shared_column = self.blocks[-1].attn.proj.weight[:, 1]
            target_mean = 1.0 / math.sqrt(cfg.d_model)
            normalized_shared_column.add_(
                target_mean - normalized_shared_column.mean()
            )
            normalized_shared_bias_column = self.blocks[-1].attn.proj.weight[:, 2]
            normalized_shared_bias_column.sub_(
                normalized_shared_bias_column.mean()
            )
>>>>>>> REPLACE

<<<<<<< SEARCH
        shared_final_scales = torch.stack(
            (
                final_proj_weight[:, 0].mean(),
                math.sqrt(self.cfg.d_model) * final_proj_weight[:, 1].mean(),
            )
        )
        x = self.ln_f(x, shared_final_scales)
=======
        shared_final_scales = torch.stack(
            (
                final_proj_weight[:, 0].mean(),
                math.sqrt(self.cfg.d_model) * final_proj_weight[:, 1].mean(),
            )
        )
        shared_final_bias = (
            math.sqrt(self.cfg.d_model) * final_proj_weight[:, 2].mean()
        )
        x = self.ln_f(x, shared_final_scales, shared_final_bias)
>>>>>>> REPLACE