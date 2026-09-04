MECHANISM: Independent-head query-basis gauge anchoring

HYPOTHESIS: Fixing `qkv.weight[4,0]` in the previously unconstrained second attention head will reduce the model to 1583 parameters while retaining at least 99% accuracy, because each head has an independent query-key basis gauge and the unsuccessful anchors constrained additional structure within the first head.

INTENDED_EDIT: Store 189 learned QKV weights and reconstruct flattened weights 0, 2, and 32 as zeros while preserving ordinary initialization random-number consumption.

EVIDENCE: Anchoring weights 0 and 2 in head 0 achieved 100% accuracy, whereas anchoring row 1 of that same head reached 97.17% and adding a third same-row constraint reached 41.31%; testing the untouched head isolates an independent gauge redundancy.

<<<<<<< SEARCH
class AttentionWeightAnchoredLinear(nn.Module):
    """QKV projection with two query weights fixed by query-key basis gauge."""
=======
class AttentionWeightAnchoredLinear(nn.Module):
    """QKV projection with three weights fixed across independent head gauges."""
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.weight_rest = nn.Parameter(torch.empty(out_features * in_features - 2))
=======
        self.weight_rest = nn.Parameter(torch.empty(out_features * in_features - 3))
>>>>>>> REPLACE

<<<<<<< SEARCH
        return torch.cat(
            (zero, self.weight_rest[:1], zero, self.weight_rest[1:])
        ).view(self.out_features, self.in_features)
=======
        return torch.cat(
            (
                zero,
                self.weight_rest[:1],
                zero,
                self.weight_rest[1:30],
                zero,
                self.weight_rest[30:],
            )
        ).view(self.out_features, self.in_features)
>>>>>>> REPLACE

<<<<<<< SEARCH
                module.weight_rest.copy_(torch.cat((flat[1:2], flat[3:])))
=======
                module.weight_rest.copy_(
                    torch.cat((flat[1:2], flat[3:32], flat[33:]))
                )
>>>>>>> REPLACE