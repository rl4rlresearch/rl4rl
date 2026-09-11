MECHANISM: Variance-normalized null-direction parameter reuse

HYPOTHESIS: Reusing attention-output column 1’s LayerNorm-invisible sum, normalized by √8, as final-LayerNorm scale coordinate 2 will reduce the model from 1,268 to 1,267 parameters while retaining at least 99% accuracy, because it preserves the function class while giving the shared scale the same effective gradient step size as a dedicated scalar.

INTENDED_EDIT: Remove the second dedicated final-LayerNorm scale, derive it from the normalized sum of projection column 1, and initialize that null component so the scale starts at one.

EVIDENCE: The first column-mean reuse achieved 100% at 1,268 parameters, while the second mean-based reuse failed at 1,267. A column mean dilutes the scale’s effective AdamW gradient across eight coefficients; normalized-sum sharing directly tests whether conditioning caused that failure.

<<<<<<< SEARCH
class FixedBiasLayerNorm(nn.Module):
    """LayerNorm with one trainable scale stored in an attention null direction."""

    def __init__(self, d_model: int):
        super().__init__()
        self.normalized_shape = (d_model,)
        self.weight = nn.Parameter(torch.ones(d_model - 6))
        self.bias = nn.Parameter(torch.zeros(d_model - 7))

    def forward(
        self, x: torch.Tensor, shared_scale: torch.Tensor
    ) -> torch.Tensor:
        weight = torch.cat(
            (
                self.weight[:1],
                self.weight.new_ones(1),
                self.weight[1:2],
                self.weight.new_ones(2),
                shared_scale.reshape(1),
                self.weight.new_ones(2),
            )
        )
=======
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
                self.weight,
                self.weight.new_ones(1),
                shared_scales[1:2],
                self.weight.new_ones(2),
                shared_scales[:1],
                self.weight.new_ones(2),
            )
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
        with torch.no_grad():
            shared_column = self.blocks[-1].attn.proj.weight[:, 0]
            shared_column.add_(1.0 - shared_column.mean())
=======
        with torch.no_grad():
            shared_weight = self.blocks[-1].attn.proj.weight
            first_shared_column = shared_weight[:, 0]
            first_shared_column.add_(1.0 - first_shared_column.mean())
            second_shared_column = shared_weight[:, 1]
            target_mean = 1.0 / math.sqrt(cfg.d_model)
            second_shared_column.add_(target_mean - second_shared_column.mean())
>>>>>>> REPLACE

<<<<<<< SEARCH
        shared_final_scale = self.blocks[-1].attn.proj.weight[:, 0].mean()
        x = self.ln_f(x, shared_final_scale)
=======
        shared_weight = self.blocks[-1].attn.proj.weight
        shared_final_scales = torch.stack(
            (
                shared_weight[:, 0].mean(),
                shared_weight[:, 1].sum() / math.sqrt(self.cfg.d_model),
            )
        )
        x = self.ln_f(x, shared_final_scales)
>>>>>>> REPLACE