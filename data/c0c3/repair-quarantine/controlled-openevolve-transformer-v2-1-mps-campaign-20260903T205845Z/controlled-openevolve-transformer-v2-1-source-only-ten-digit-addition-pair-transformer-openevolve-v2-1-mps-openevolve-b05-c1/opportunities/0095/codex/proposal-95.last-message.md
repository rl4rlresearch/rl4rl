MECHANISM: Fused dual null-direction parameter reuse

HYPOTHESIS: Reusing a second LayerNorm-invisible attention-output column mean will produce a 1,267-parameter model with at least 99% accuracy; computing both shared scales in one reduction may avoid the prior attempt’s timeout.

INTENDED_EDIT: Remove final-LayerNorm scale coordinate 2, derive it from attention projection column 1, initialize both shared column means to one, and extract them together.

EVIDENCE: The first null-direction reuse achieved 100% accuracy at 1,268 parameters. The analogous second reuse only timed out, providing no evidence of an accuracy failure, while fixed-scale pruning failed at 16.73%.

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