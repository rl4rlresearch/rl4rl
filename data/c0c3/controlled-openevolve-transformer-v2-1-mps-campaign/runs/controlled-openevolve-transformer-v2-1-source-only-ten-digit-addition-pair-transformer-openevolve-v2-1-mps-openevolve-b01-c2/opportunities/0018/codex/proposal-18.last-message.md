MECHANISM: Second-row LayerNorm-nullspace weight gauge fixing

HYPOTHESIS: Removing the redundant final `fc1` input-weight coordinate from a second output row will reduce the model to 1,608 parameters while retaining at least 99% accuracy, because every row independently acts on the same zero-mean normalized vector and the first-row gauge achieved 99.97%.

INTENDED_EDIT: Gauge-fix the final input weight in each of the first two `fc1` rows, synthesizing both as zero and gauge-adjusting both rows during initialization.

EVIDENCE: The verified 1,609-parameter design successfully removed one `fc1` coordinate with 99.97% accuracy; extending that exact row-local nullspace reduction by one row is the smallest and most directly supported next change.

<<<<<<< SEARCH
class LayerNormGaugedLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # Consume the same constructor RNG stream as the replaced nn.Linear.
        base = nn.Linear(in_features, out_features)
        self.weight = nn.Parameter(base.weight.new_empty(out_features * in_features - 1))
        self.bias = nn.Parameter(base.bias.new_empty(out_features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        anchor = self.in_features - 1
        weight = torch.cat(
            (self.weight[:anchor], self.weight.new_zeros(1), self.weight[anchor:])
        ).view(self.out_features, self.in_features)
        return F.linear(x, weight, self.bias)
=======
class LayerNormGaugedLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # Consume the same constructor RNG stream as the replaced nn.Linear.
        base = nn.Linear(in_features, out_features)
        self.weight = nn.Parameter(base.weight.new_empty(out_features * in_features - 2))
        self.bias = nn.Parameter(base.bias.new_empty(out_features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        row_width = self.in_features - 1
        weight = torch.cat(
            (
                self.weight[:row_width],
                self.weight.new_zeros(1),
                self.weight[row_width : 2 * row_width],
                self.weight.new_zeros(1),
                self.weight[2 * row_width :],
            )
        ).view(self.out_features, self.in_features)
        return F.linear(x, weight, self.bias)
>>>>>>> REPLACE

<<<<<<< SEARCH
                # For normalized z, sum(z) is zero. Subtracting the omitted
                # coefficient from the other row coefficients preserves row 0.
                full[0, :-1].sub_(full[0, -1].clone())
                flat = full.flatten()
                anchor = module.in_features - 1
                module.weight.copy_(torch.cat((flat[:anchor], flat[anchor + 1 :])))
                nn.init.zeros_(module.bias)
=======
                # For normalized z, sum(z) is zero. Each output row therefore
                # has an independent common-coefficient null direction.
                full[:2, :-1].sub_(full[:2, -1:].clone())
                module.weight.copy_(
                    torch.cat((full[:2, :-1].flatten(), full[2:].flatten()))
                )
                nn.init.zeros_(module.bias)
>>>>>>> REPLACE