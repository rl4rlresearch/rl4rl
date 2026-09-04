MECHANISM: Nonzero query-key scale-gauge anchoring

HYPOTHESIS: Fixing `qkv.weight[0,4]` at 0.02 while inversely rescaling head-0 key row 0 will reduce the model to 1583 parameters and retain at least 99% accuracy, because this consumes the remaining query-key scale gauge without eliminating another query input direction.

INTENDED_EDIT: Store 189 learned QKV weights, reconstruct weights 0 and 2 as zeros and weight 4 as 0.02, and initialize the corresponding query/key rows with reciprocal scaling that preserves the successful 1584-parameter model’s initial attention scores.

EVIDENCE: The two-zero layout reached 100% at 1584 parameters, whereas making weight 4 a third zero reached 41.31% and function-preserving orthogonal initialization only recovered 73.56%. A nonzero anchor preserves weight 4’s contribution while using scale rather than another directional constraint.

<<<<<<< SEARCH
class AttentionWeightAnchoredLinear(nn.Module):
    """QKV projection with two query weights fixed by query-key basis gauge."""

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # Preserve the random-number consumption of nn.Linear construction.
        nn.Linear(in_features, out_features, bias=False)
        self.weight_rest = nn.Parameter(torch.empty(out_features * in_features - 2))

    def weight_tensor(self) -> torch.Tensor:
        zero = self.weight_rest.new_zeros(1)
        return torch.cat(
            (zero, self.weight_rest[:1], zero, self.weight_rest[1:])
        ).view(self.out_features, self.in_features)
=======
class AttentionWeightAnchoredLinear(nn.Module):
    """QKV projection with two zero anchors and one query scale anchor."""

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # Preserve the random-number consumption of nn.Linear construction.
        nn.Linear(in_features, out_features, bias=False)
        self.weight_rest = nn.Parameter(torch.empty(out_features * in_features - 3))

    def weight_tensor(self) -> torch.Tensor:
        zero = self.weight_rest.new_zeros(1)
        fixed = self.weight_rest.new_full((1,), 0.02)
        return torch.cat(
            (
                zero,
                self.weight_rest[:1],
                zero,
                self.weight_rest[1:2],
                fixed,
                self.weight_rest[2:],
            )
        ).view(self.out_features, self.in_features)
>>>>>>> REPLACE

<<<<<<< SEARCH
        if isinstance(module, AttentionWeightAnchoredLinear):
            weight = module.weight_rest.new_empty(
                module.out_features, module.in_features
            )
            nn.init.normal_(weight, mean=0.0, std=0.02)
            with torch.no_grad():
                flat = weight.flatten()
                module.weight_rest.copy_(torch.cat((flat[1:2], flat[3:])))
=======
        if isinstance(module, AttentionWeightAnchoredLinear):
            weight = module.weight_rest.new_empty(
                module.out_features, module.in_features
            )
            nn.init.normal_(weight, mean=0.0, std=0.02)
            with torch.no_grad():
                weight[0, 0] = 0.0
                weight[0, 2] = 0.0

                # Fix one nonzero query coefficient using the Q/K reciprocal
                # scale gauge, preserving this head coordinate's attention.
                scale = 0.02 / weight[0, 4]
                weight[0].mul_(scale)
                weight[8].div_(scale)

                flat = weight.flatten()
                module.weight_rest.copy_(
                    torch.cat((flat[1:2], flat[3:4], flat[5:]))
                )
>>>>>>> REPLACE