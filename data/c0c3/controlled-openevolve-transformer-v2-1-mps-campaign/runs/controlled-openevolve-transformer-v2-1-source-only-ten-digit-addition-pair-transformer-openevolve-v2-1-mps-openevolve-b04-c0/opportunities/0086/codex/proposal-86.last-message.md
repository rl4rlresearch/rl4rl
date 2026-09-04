MECHANISM: Two-column query-row gauge anchoring

HYPOTHESIS: Fixing `qkv.weight[0,2]` alongside `qkv.weight[0,0]` will reduce the model to 1584 parameters while retaining at least 99% accuracy, because a query-basis covector can be oriented orthogonally to two input-column vectors without suppressing a second query coordinate.

INTENDED_EDIT: Reconstruct flattened QKV weights 0 and 2 as zeros while learning the other 190 weights and preserving ordinary initialization random-number consumption.

EVIDENCE: The first query anchor achieved 99.55% at 1585 parameters, whereas anchoring a second query coordinate on the same input column reached only 97.17%; anchoring a second input column on the already-constrained query coordinate tests a distinct gauge layout, and input coordinate 2 has already tolerated a fixed first-LayerNorm gain.

<<<<<<< SEARCH
class AttentionWeightAnchoredLinear(nn.Module):
    """QKV projection with one query weight fixed by query-key basis gauge."""

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # Preserve the random-number consumption of nn.Linear construction.
        nn.Linear(in_features, out_features, bias=False)
        self.weight_rest = nn.Parameter(torch.empty(out_features * in_features - 1))

    def weight_tensor(self) -> torch.Tensor:
        zero = self.weight_rest.new_zeros(1)
        return torch.cat((zero, self.weight_rest)).view(
            self.out_features, self.in_features
        )
=======
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
>>>>>>> REPLACE

<<<<<<< SEARCH
            with torch.no_grad():
                module.weight_rest.copy_(weight.flatten()[1:])
=======
            with torch.no_grad():
                flat = weight.flatten()
                module.weight_rest.copy_(torch.cat((flat[1:2], flat[3:])))
>>>>>>> REPLACE