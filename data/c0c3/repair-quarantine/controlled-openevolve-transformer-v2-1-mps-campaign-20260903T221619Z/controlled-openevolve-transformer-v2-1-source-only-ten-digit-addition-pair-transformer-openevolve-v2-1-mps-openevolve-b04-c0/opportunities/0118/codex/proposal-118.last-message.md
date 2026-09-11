MECHANISM: MLP output residual-shift weight gauge

HYPOTHESIS: Fixing `fc2.weight[0,0]` at zero will produce a 1567-parameter model with at least 99% accuracy, because subtracting this coefficient from every output row adds only a feature-uniform residual shift that the final LayerNorm removes.

INTENDED_EDIT: Replace the MLP output layer’s 96-weight matrix with 95 learned weights, reconstruct `fc2.weight[0,0]` as zero, and canonicalize its fresh initialization while retaining the existing anchored bias.

EVIDENCE: The identical residual-shift canonicalization for `proj.weight[0,0]` achieved 99.93% accuracy at 1569 parameters; applying the first such anchor to the independent MLP residual branch is better supported than adding another projection-column anchor after `[0,1]` failed.

<<<<<<< SEARCH
class FinalBiasAnchoredLinear(nn.Linear):
    """Linear layer with bias coordinate 1 fixed at zero."""

    def __init__(self, in_features: int, out_features: int):
        # Construct the ordinary bias first to preserve initialization RNG use.
        super().__init__(in_features, out_features, bias=True)
        self.bias = None
        self.bias_rest = nn.Parameter(torch.zeros(out_features - 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        zero = self.bias_rest.new_zeros(1)
        bias = torch.cat((self.bias_rest[:1], zero, self.bias_rest[1:]))
        return F.linear(x, self.weight, bias)
=======
class FinalBiasAnchoredLinear(nn.Module):
    """MLP output projection with one residual-gauge weight and bias anchor."""

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # Preserve the random-number consumption of nn.Linear construction.
        nn.Linear(in_features, out_features, bias=True)
        self.weight_rest = nn.Parameter(
            torch.empty(out_features * in_features - 1)
        )
        self.bias_rest = nn.Parameter(torch.zeros(out_features - 1))

    def weight_tensor(self) -> torch.Tensor:
        zero = self.weight_rest.new_zeros(1)
        return torch.cat((zero, self.weight_rest)).view(
            self.out_features, self.in_features
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        zero = self.bias_rest.new_zeros(1)
        bias = torch.cat((self.bias_rest[:1], zero, self.bias_rest[1:]))
        return F.linear(x, self.weight_tensor(), bias)
>>>>>>> REPLACE

<<<<<<< SEARCH
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        elif isinstance(module, SharedAnchorEmbeddings):
=======
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        elif isinstance(module, FinalBiasAnchoredLinear):
            weight = module.weight_rest.new_empty(
                module.out_features, module.in_features
            )
            nn.init.normal_(weight, mean=0.0, std=0.02)
            with torch.no_grad():
                # Removing this coefficient from every output row changes the
                # residual stream only by a feature-uniform, tokenwise shift.
                offset = weight[0, 0].clone()
                weight[:, 0].sub_(offset)
                weight[0, 0] = 0.0
                module.weight_rest.copy_(weight.flatten()[1:])
                nn.init.zeros_(module.bias_rest)
        elif isinstance(module, SharedAnchorEmbeddings):
>>>>>>> REPLACE