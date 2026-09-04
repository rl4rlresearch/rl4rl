MECHANISM: Attention null-direction reuse for final normalization bias

HYPOTHESIS: Reusing a second LayerNorm-invisible attention-output column mean as the learned final-LayerNorm bias will reduce the verified model from 1,268 to 1,267 parameters while retaining at least 99% accuracy, because it preserves both the full function class and the freshly initialized function.

INTENDED_EDIT: Remove the dedicated final-LayerNorm bias, derive it from attention projection column 1, and initialize that column’s mean to zero.

EVIDENCE: Reusing projection column 0’s null-direction mean as a final-LayerNorm scale retained 100% accuracy at 1,268 parameters. Unlike the timed-out attempt to reuse another scale, this tests the same proven symmetry on the independently initialized zero-valued normalization bias.

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
        return F.layer_norm(
            x,
            self.normalized_shape,
            weight,
            F.pad(self.bias, (0, 7)),
        )
=======
class FixedBiasLayerNorm(nn.Module):
    """LayerNorm with trainable affine values stored in attention null directions."""

    def __init__(self, d_model: int):
        super().__init__()
        self.normalized_shape = (d_model,)
        self.weight = nn.Parameter(torch.ones(d_model - 6))

    def forward(
        self,
        x: torch.Tensor,
        shared_scale: torch.Tensor,
        shared_bias: torch.Tensor,
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
        return F.layer_norm(
            x,
            self.normalized_shape,
            weight,
            F.pad(shared_bias.reshape(1), (0, 7)),
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
        with torch.no_grad():
            shared_column = self.blocks[-1].attn.proj.weight[:, 0]
            shared_column.add_(1.0 - shared_column.mean())
        nn.init.normal_(self.token_emb, mean=0.0, std=0.02)
=======
        with torch.no_grad():
            shared_column = self.blocks[-1].attn.proj.weight[:, 0]
            shared_column.add_(1.0 - shared_column.mean())
            shared_bias_column = self.blocks[-1].attn.proj.weight[:, 1]
            shared_bias_column.sub_(shared_bias_column.mean())
        nn.init.normal_(self.token_emb, mean=0.0, std=0.02)
>>>>>>> REPLACE

<<<<<<< SEARCH
        shared_final_scale = self.blocks[-1].attn.proj.weight[:, 0].mean()
        x = self.ln_f(x, shared_final_scale)
=======
        shared_final_scale = self.blocks[-1].attn.proj.weight[:, 0].mean()
        shared_final_bias = self.blocks[-1].attn.proj.weight[:, 1].mean()
        x = self.ln_f(x, shared_final_scale, shared_final_bias)
>>>>>>> REPLACE