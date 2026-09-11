MECHANISM: Null-direction parameter reuse

HYPOTHESIS: Reusing a LayerNorm-invisible row-common attention-output coefficient as the sixth final-LayerNorm scale will reduce the model from 1,269 to 1,268 parameters while retaining at least 99% accuracy, because the reparameterization preserves the full function class and fresh initialization function.

INTENDED_EDIT: Remove one dedicated final-LayerNorm scale and derive it from the mean of the final attention projection’s first column, whose row-common component is otherwise erased by subsequent LayerNorms.

EVIDENCE: Fixing this final-LayerNorm scale at one reached only 16.73%, showing that it must remain trainable; directly removing a row-common projection degree also failed. Reusing that null direction preserves both degrees of functional freedom while the current model’s 1,269-parameter configuration reached 100%.

<<<<<<< SEARCH
class FixedBiasLayerNorm(nn.Module):
    """LayerNorm with five fixed scales and seven fixed bias coordinates."""

    def __init__(self, d_model: int):
        super().__init__()
        self.normalized_shape = (d_model,)
        self.weight = nn.Parameter(torch.ones(d_model - 5))
        self.bias = nn.Parameter(torch.zeros(d_model - 7))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = torch.cat(
            (
                self.weight[:1],
                self.weight.new_ones(1),
                self.weight[1:2],
                self.weight.new_ones(2),
                self.weight[2:],
                self.weight.new_ones(2),
            )
        )
=======
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
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.apply(self._init_weights)
        nn.init.normal_(self.token_emb, mean=0.0, std=0.02)
=======
        self.apply(self._init_weights)
        with torch.no_grad():
            shared_column = self.blocks[-1].attn.proj.weight[:, 0]
            shared_column.add_(1.0 - shared_column.mean())
        nn.init.normal_(self.token_emb, mean=0.0, std=0.02)
>>>>>>> REPLACE

<<<<<<< SEARCH
        x = self.ln_f(x)
        logits = F.linear(x, self.token_weight())
=======
        shared_final_scale = self.blocks[-1].attn.proj.weight[:, 0].mean()
        x = self.ln_f(x, shared_final_scale)
        logits = F.linear(x, self.token_weight())
>>>>>>> REPLACE