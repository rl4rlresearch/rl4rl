MECHANISM: Function-preserving third query-basis gauge anchor

HYPOTHESIS: Reducing the model to 1583 parameters by fixing `qkv.weight[0,4]` will retain at least 99% accuracy when initialization orthogonally transforms both head-0 query and key bases, preserving their initial attention scores instead of simply deleting a random query coefficient.

INTENDED_EDIT: Fix flattened QKV weights 0, 2, and 4, store the remaining 189 weights, and initialize the constrained head through a joint orthogonal query-key basis transformation.

EVIDENCE: The two-anchor layout at weights 0 and 2 achieved 100% accuracy, while naively adding weight 4 fell to 41.31%; preserving the unconstrained initialization’s attention function directly tests whether that failure arose from the uncompensated initialization disturbance rather than the gauge constraint itself.

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
    """QKV projection with three query weights fixed by query-key basis gauge."""

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # Preserve the random-number consumption of nn.Linear construction.
        nn.Linear(in_features, out_features, bias=False)
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

            # Choose an orthogonal head-0 basis whose first query row is
            # perpendicular to input columns 0, 2, and 4. Applying the same
            # basis to its keys preserves every initial attention score.
            query_columns = weight[:4, (0, 2, 4)].T
            _u, _s, vh = torch.linalg.svd(query_columns, full_matrices=True)
            basis = torch.cat((vh[-1:], vh[:-1]), dim=0)
            weight[:4] = basis @ weight[:4]
            weight[8:12] = basis @ weight[8:12]

            with torch.no_grad():
                flat = weight.flatten()
                module.weight_rest.copy_(
                    torch.cat((flat[1:2], flat[3:4], flat[5:]))
                )
>>>>>>> REPLACE