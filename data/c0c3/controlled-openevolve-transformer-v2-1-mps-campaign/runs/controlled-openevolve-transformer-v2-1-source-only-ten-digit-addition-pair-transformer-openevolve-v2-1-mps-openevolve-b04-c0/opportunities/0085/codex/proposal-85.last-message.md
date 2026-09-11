MECHANISM: Second query-key basis gauge anchor

HYPOTHESIS: Fixing `qkv.weight[1,0]` alongside `qkv.weight[0,0]` will reduce the model to 1584 parameters while retaining at least 99% accuracy, because an invertible basis transformation can orient the four-dimensional head-0 query column into the remaining two coordinates without changing attention scores.

INTENDED_EDIT: Store 190 learned QKV weights, reconstruct flattened QKV weights 0 and 8 as zeros, and preserve ordinary initialization random-number consumption for every learned weight.

EVIDENCE: The first query-weight anchor achieved 99.55% at 1585 parameters, while the attempted LayerNorm-gain reduction reached only 98.53%; extending the successful query-key gauge mechanism is therefore the most informative next reduction.

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
        flat = torch.cat(
            (zero, self.weight_rest[: self.in_features - 1], zero,
             self.weight_rest[self.in_features - 1 :])
        )
        return flat.view(self.out_features, self.in_features)
>>>>>>> REPLACE

<<<<<<< SEARCH
            with torch.no_grad():
                module.weight_rest.copy_(weight.flatten()[1:])
=======
            flat = weight.flatten()
            with torch.no_grad():
                module.weight_rest.copy_(
                    torch.cat(
                        (
                            flat[1 : module.in_features],
                            flat[module.in_features + 1 :],
                        )
                    )
                )
>>>>>>> REPLACE