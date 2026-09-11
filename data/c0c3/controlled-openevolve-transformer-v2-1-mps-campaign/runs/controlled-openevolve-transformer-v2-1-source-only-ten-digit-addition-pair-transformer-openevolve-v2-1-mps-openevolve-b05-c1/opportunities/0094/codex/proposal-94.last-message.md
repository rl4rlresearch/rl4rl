MECHANISM: Second null-direction parameter reuse

HYPOTHESIS: Reusing a second LayerNorm-invisible attention-output column mean as the remaining learned final-LayerNorm scale will reduce the model from 1,268 to 1,267 parameters while retaining at least 99% accuracy, because it preserves both the full function class and the fresh initialization function.

INTENDED_EDIT: Remove the dedicated final-LayerNorm scale for coordinate 2, derive it from the mean of attention projection column 1, and initialize that column mean to one.

EVIDENCE: The first null-direction reuse retained 100% accuracy at 1,268 parameters, while fixing a final-LayerNorm scale reached only 16.73%; duplicating the successful reuse with an independent null projection direction preserves the required trainability.

<<<<<<< SEARCH
class FixedBiasLayerNorm(nn.Module):
    """LayerNorm with one trainable scale stored in an attention null direction."""

    def __init__(self, d_model: int):
        super().__init__()
        self.normalized_shape = (d_model,)
        self.weight = nn.Parameter(torch.ones(d_model - 6))
        self.bias = nn.Parameter(torch.zeros(d_model - 7))
=======
class FixedBiasLayerNorm(nn.Module):
    """LayerNorm with two trainable scales stored in attention null directions."""

    def __init__(self, d_model: int):
        super().__init__()
        self.normalized_shape = (d_model,)
        self.weight = nn.Parameter(torch.ones(d_model - 7))
        self.bias = nn.Parameter(torch.zeros(d_model - 7))
>>>>>>> REPLACE

<<<<<<< SEARCH
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
    def forward(
        self, x: torch.Tensor, shared_scales: torch.Tensor
    ) -> torch.Tensor:
        shared_scales = shared_scales.reshape(-1)
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
>>>>>>> REPLACE

<<<<<<< SEARCH
        with torch.no_grad():
            shared_column = self.blocks[-1].attn.proj.weight[:, 0]
            shared_column.add_(1.0 - shared_column.mean())
=======
        with torch.no_grad():
            shared_columns = self.blocks[-1].attn.proj.weight[:, :2]
            shared_columns.add_(
                1.0 - shared_columns.mean(dim=0, keepdim=True)
            )
>>>>>>> REPLACE

<<<<<<< SEARCH
        shared_final_scale = self.blocks[-1].attn.proj.weight[:, 0].mean()
        x = self.ln_f(x, shared_final_scale)
=======
        shared_final_scales = self.blocks[-1].attn.proj.weight[:, :2].mean(dim=0)
        x = self.ln_f(x, shared_final_scales)
>>>>>>> REPLACE