MECHANISM: Null-direction reuse for final-LayerNorm bias

HYPOTHESIS: Reusing attention-output column 1’s LayerNorm-invisible mean as the remaining learned final-LayerNorm bias will reduce the verified model from 1,268 to 1,267 parameters while retaining at least 99% accuracy.

INTENDED_EDIT: Remove the dedicated final-LayerNorm bias parameter, derive that bias from projection column 1’s mean, and center the column at initialization so the fresh model still starts with zero final bias.

EVIDENCE: The first projection-column mean reuse retained 100% accuracy at 1,268 parameters, establishing this null-direction sharing mechanism. Both attempts to reuse a second scale failed, motivating reuse of the distinct additive affine degree of freedom instead; seven of eight final biases are already fixed, and the sole learned bias remains fully trainable under this exact reparameterization.

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
=======
class FixedBiasLayerNorm(nn.Module):
    """LayerNorm with affine scalars stored in attention null directions."""

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
>>>>>>> REPLACE

<<<<<<< SEARCH
        return F.layer_norm(
            x,
            self.normalized_shape,
            weight,
            F.pad(self.bias, (0, 7)),
        )
=======
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
        shared_affine = self.blocks[-1].attn.proj.weight[:, :2].mean(dim=0)
        x = self.ln_f(x, shared_affine[0], shared_affine[1])
>>>>>>> REPLACE