MECHANISM: Zero-mean LayerNorm incoming-weight gauge anchoring

HYPOTHESIS: Fixing `fc1.weight[8,0]` at zero will reduce the model from 1599 to 1598 parameters while retaining at least 99% accuracy, because LayerNorm’s normalized coordinates sum to zero and the still-learned hidden-unit-8 bias preserves the corresponding affine function class.

INTENDED_EDIT: Replace the full `fc1` weight with 95 learned coordinates, reconstruct `weight[8,0]` as zero, and gauge-transform its ordinary initialization without changing the initial model function.

EVIDENCE: Removing `fc1.bias[8]` reduced accuracy to 77.57%, showing that bias should remain learned; the 1599-parameter design reached 99.91%, motivating removal of an exact incoming-weight redundancy from that same unit instead.

<<<<<<< SEARCH
class BiasAnchoredLinear(nn.Linear):
    """Linear layer with bias coordinates 0 through 7 fixed at zero."""

    def __init__(self, in_features: int, out_features: int):
        # Construct the ordinary bias first to preserve initialization RNG use.
        super().__init__(in_features, out_features, bias=True)
        self.bias = None
        self.bias_rest = nn.Parameter(torch.zeros(out_features - 8))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        zeros = self.bias_rest.new_zeros(8)
        bias = torch.cat((zeros, self.bias_rest))
        return F.linear(x, self.weight, bias)
=======
class BiasAnchoredLinear(nn.Linear):
    """Linear layer with eight fixed biases and one zero-mean input gauge fixed."""

    def __init__(self, in_features: int, out_features: int):
        # Construct the ordinary layer first to preserve initialization RNG use.
        super().__init__(in_features, out_features, bias=True)
        self.weight = None
        self.bias = None
        self.weight_rest = nn.Parameter(torch.empty(out_features * in_features - 1))
        self.bias_rest = nn.Parameter(torch.zeros(out_features - 8))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        anchor_index = 8 * self.in_features
        zero = self.weight_rest.new_zeros(1)
        weight = torch.cat(
            (self.weight_rest[:anchor_index], zero, self.weight_rest[anchor_index:])
        ).view(self.out_features, self.in_features)
        bias = torch.cat((self.bias_rest.new_zeros(8), self.bias_rest))
        return F.linear(x, weight, bias)
>>>>>>> REPLACE

<<<<<<< SEARCH
    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, SharedAnchorEmbeddings):
=======
    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, BiasAnchoredLinear):
            weight = module.weight_rest.new_empty(
                module.out_features, module.in_features
            )
            nn.init.normal_(weight, mean=0.0, std=0.02)

            # At initialization ln2 has unit gain and zero shift, so subtracting
            # one row's first coefficient from that entire row is function
            # preserving because its normalized inputs sum to zero.
            anchored = weight.clone()
            anchored[8] = weight[8] - weight[8, 0]
            anchor_index = 8 * module.in_features
            flat = anchored.flatten()
            with torch.no_grad():
                module.weight_rest.copy_(
                    torch.cat((flat[:anchor_index], flat[anchor_index + 1 :]))
                )
                module.bias_rest.zero_()
        elif isinstance(module, SharedAnchorEmbeddings):
>>>>>>> REPLACE