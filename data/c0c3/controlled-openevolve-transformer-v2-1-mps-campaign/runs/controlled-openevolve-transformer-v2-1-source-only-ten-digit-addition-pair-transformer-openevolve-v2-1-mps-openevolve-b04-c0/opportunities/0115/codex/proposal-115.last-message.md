MECHANISM: Attention-output residual-shift weight gauge

HYPOTHESIS: Fixing `proj.weight[0,0]` at zero will produce a 1569-parameter model with at least 99% accuracy, because subtracting that coefficient from every output row changes each token only by a feature-uniform residual shift, which downstream LayerNorms remove under the fixed zero-dropout configuration.

INTENDED_EDIT: Replace the attention output projection with a 63-weight parameterization, reconstruct `proj.weight[0,0]` as zero, and canonicalize its initialization while leaving the tied value/projection bias unchanged.

EVIDENCE: The current 1570-parameter design reached 99.46%, whereas constraining the tied bias reached only 19.94%; this tests the independent projection-weight residual gauge without perturbing the bias coupling implicated by that failure.

<<<<<<< SEARCH
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight_tensor())


class CausalSelfAttention(nn.Module):
=======
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight_tensor())


class ResidualGaugeLinear(nn.Module):
    """Projection with one weight fixed along a feature-uniform residual gauge."""

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # Preserve the random-number consumption of nn.Linear construction.
        nn.Linear(in_features, out_features, bias=True)
        self.weight_rest = nn.Parameter(
            torch.empty(out_features * in_features - 1)
        )
        self.bias = None

    def weight_tensor(self) -> torch.Tensor:
        zero = self.weight_rest.new_zeros(1)
        return torch.cat((zero, self.weight_rest)).view(
            self.out_features, self.in_features
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight_tensor(), self.bias)


class CausalSelfAttention(nn.Module):
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.proj = nn.Linear(d_model, d_model)
=======
        self.proj = ResidualGaugeLinear(d_model, d_model)
>>>>>>> REPLACE

<<<<<<< SEARCH
                            flat[100:],
                        )
                    )
                )
        elif isinstance(module, SharedAnchorEmbeddings):
=======
                            flat[100:],
                        )
                    )
                )
        elif isinstance(module, ResidualGaugeLinear):
            weight = module.weight_rest.new_empty(
                module.out_features, module.in_features
            )
            nn.init.normal_(weight, mean=0.0, std=0.02)
            with torch.no_grad():
                # Subtracting one input-column coefficient from every output
                # row adds only a feature-uniform residual shift. Subsequent
                # LayerNorms remove that shift exactly.
                offset = weight[0, 0].clone()
                weight[:, 0].sub_(offset)
                weight[0, 0] = 0.0
                module.weight_rest.copy_(weight.flatten()[1:])
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        elif isinstance(module, SharedAnchorEmbeddings):
>>>>>>> REPLACE