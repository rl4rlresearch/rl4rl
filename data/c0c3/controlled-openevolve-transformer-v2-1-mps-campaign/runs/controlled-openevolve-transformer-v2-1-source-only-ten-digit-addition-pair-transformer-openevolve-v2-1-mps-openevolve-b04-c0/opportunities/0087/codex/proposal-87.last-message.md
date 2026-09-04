MECHANISM: Third same-row query-basis gauge anchor

HYPOTHESIS: Fixing `qkv.weight[0,4]` alongside `qkv.weight[0,0]` and `qkv.weight[0,2]` will reduce the model to 1583 parameters while retaining at least 99% accuracy, because a four-dimensional query-basis covector can be oriented orthogonally to three input-column vectors.

INTENDED_EDIT: Store 189 learned QKV weights, reconstruct flattened QKV weights 0, 2, and 4 as zeros, and preserve ordinary initialization random-number consumption.

EVIDENCE: Anchoring weights 0 and 2 in the same query row achieved 100% accuracy at 1584 parameters, whereas anchoring a second query coordinate on input column 0 reached only 97.17%; extending the successful same-row layout to coordinate 4 is the most direct capacity test.

<<<<<<< SEARCH
class AttentionWeightAnchoredLinear(nn.Module):
    """QKV projection with two query weights fixed by query-key basis gauge."""
=======
class AttentionWeightAnchoredLinear(nn.Module):
    """QKV projection with three query weights fixed by query-key basis gauge."""
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.weight_rest = nn.Parameter(torch.empty(out_features * in_features - 2))

    def weight_tensor(self) -> torch.Tensor:
        zero = self.weight_rest.new_zeros(1)
        return torch.cat(
            (zero, self.weight_rest[:1], zero, self.weight_rest[1:])
        ).view(self.out_features, self.in_features)
=======
        self.weight_rest = nn.Parameter(torch.empty(out_features * in_features - 3))

    def weight_tensor(self) -> torch.Tensor:
        zero = self.weight_rest.new_zeros(1)
        return torch.cat(
            (
                zero,
                self.weight_rest[:1],
                zero,
                self.weight_rest[1:2],
                zero,
                self.weight_rest[2:],
            )
        ).view(self.out_features, self.in_features)
>>>>>>> REPLACE

<<<<<<< SEARCH
                module.weight_rest.copy_(torch.cat((flat[1:2], flat[3:])))
=======
                module.weight_rest.copy_(
                    torch.cat((flat[1:2], flat[3:4], flat[5:]))
                )
>>>>>>> REPLACE