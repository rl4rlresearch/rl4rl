MECHANISM: Orthonormal attention-null parameter reuse

HYPOTHESIS: Reusing the normalized row-common component of attention projection column 1 as the second learned final-LayerNorm scale will reduce the model from 1,268 to 1,267 parameters while retaining at least 99% accuracy; unlike the timed-out raw-mean attempt, the orthonormal scaling gives this scale unit gradient norm and preserves the initialized function exactly.

INTENDED_EDIT: Remove the dedicated final-LayerNorm scale for coordinate 2, derive it as the normalized mean of projection column 1, and initialize that null component to produce scale one.

EVIDENCE: The first projection-null reuse retained 100% accuracy at 1,268 parameters, establishing that row-common attention-output components can carry required normalization degrees of freedom. The prior second-scale attempt produced no accuracy result because training timed out, so testing the same exact symmetry with an orthonormally conditioned coordinate is the closest informative reduction.

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
        nn.init.normal_(self.token_emb, mean=0.0, std=0.02)
=======
        with torch.no_grad():
            shared_column = self.blocks[-1].attn.proj.weight[:, 0]
            shared_column.add_(1.0 - shared_column.mean())
            normalized_shared_column = self.blocks[-1].attn.proj.weight[:, 1]
            target_mean = 1.0 / math.sqrt(cfg.d_model)
            normalized_shared_column.add_(
                target_mean - normalized_shared_column.mean()
            )
        nn.init.normal_(self.token_emb, mean=0.0, std=0.02)
>>>>>>> REPLACE

<<<<<<< SEARCH
        shared_final_scale = self.blocks[-1].attn.proj.weight[:, 0].mean()
        x = self.ln_f(x, shared_final_scale)
=======
        final_proj_weight = self.blocks[-1].attn.proj.weight
        shared_final_scales = torch.stack(
            (
                final_proj_weight[:, 0].mean(),
                math.sqrt(self.cfg.d_model) * final_proj_weight[:, 1].mean(),
            )
        )
        x = self.ln_f(x, shared_final_scales)
>>>>>>> REPLACE