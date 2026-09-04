MECHANISM: Orthonormal LayerNorm-gauge bias packing

HYPOTHESIS: The 1,269-parameter model will achieve at least 99% accuracy because it preserves the current verified model’s full MLP function class and effective zero-bias initialization while storing its three quartet biases in three otherwise inactive row-constant weight directions.

INTENDED_EDIT: Remove the three explicit MLP hidden-bias parameters and derive each quartet bias from the normalized row sum of one dense weight row; center those carrier rows at initialization so their initial biases remain zero.

EVIDENCE: The current 1,272-parameter model achieved 99.28%. The 1,237-parameter LayerNorm-subspace experiment showed that wholesale projection reparameterization disrupted training despite targeting algebraically inactive directions, motivating reuse of only three such directions while retaining the dense weight layout and its optimization geometry.

<<<<<<< SEARCH
class PairwiseTiedBiasLinear(nn.Linear):
    """Linear layer whose outputs form learned-bias quartets."""

    def __init__(self, in_features: int, out_features: int):
        if out_features % 4 != 0:
            raise ValueError("out_features must be divisible by four")
        super().__init__(in_features, out_features, bias=True)
        self.bias = None
        self.free_bias = nn.Parameter(torch.zeros(out_features // 4))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bias = self.free_bias.repeat_interleave(4)
        return F.linear(x, self.weight, bias)
=======
class PairwiseTiedBiasLinear(nn.Linear):
    """Linear layer with quartet biases packed into inactive weight directions."""

    def __init__(self, in_features: int, out_features: int):
        if out_features % 4 != 0:
            raise ValueError("out_features must be divisible by four")
        super().__init__(in_features, out_features, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        quartet_weight = self.weight.view(-1, 4, self.in_features)
        carrier_rows = quartet_weight[:, 0, :]
        quartet_bias = carrier_rows.sum(dim=-1) / math.sqrt(self.in_features)
        bias = quartet_bias.repeat_interleave(4)
        return F.linear(x, self.weight, bias)
>>>>>>> REPLACE

<<<<<<< SEARCH
    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)
=======
    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)
            if isinstance(module, PairwiseTiedBiasLinear):
                with torch.no_grad():
                    carrier_rows = module.weight.view(
                        -1, 4, module.in_features
                    )[:, 0, :]
                    carrier_rows.sub_(carrier_rows.mean(dim=-1, keepdim=True))
>>>>>>> REPLACE