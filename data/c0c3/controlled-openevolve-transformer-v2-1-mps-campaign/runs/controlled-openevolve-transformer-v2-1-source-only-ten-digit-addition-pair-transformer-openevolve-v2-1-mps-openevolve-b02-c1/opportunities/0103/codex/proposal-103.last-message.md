MECHANISM: Mid-pair final-LayerNorm scale sharing

HYPOTHESIS: Sharing final-LayerNorm scales for residual coordinates two and three will reduce the model from 985 to 984 parameters while retaining at least 99% accuracy, because those coordinates already share an MLP input coefficient and tolerate zero companion-key contributions, while the sensitive later coordinates remain independent.

INTENDED_EDIT: Store five final-LayerNorm scales instead of six and reuse the third scale for coordinates two and three, preserving the existing scale sharing for coordinates five and six.

EVIDENCE: The 989-parameter design achieved 99.98% with the first two companion-key tail coefficients—corresponding to residual coordinates two and three—fixed at zero, and disjoint MLP input pairing achieved 99.93%; unlike the failed terminal scale extension, this change leaves coordinates four through six unchanged.

<<<<<<< SEARCH
        self.weight = nn.Parameter(torch.ones(normalized_shape - 2))
        self.bias = nn.Parameter(torch.zeros(normalized_shape - 6))
=======
        self.weight = nn.Parameter(torch.ones(normalized_shape - 3))
        self.bias = nn.Parameter(torch.zeros(normalized_shape - 6))
>>>>>>> REPLACE

<<<<<<< SEARCH
        weight = torch.cat(
            [
                self.weight,
                self.weight[-1:],
                self.weight.new_ones(1),
            ]
        )
=======
        weight = torch.cat(
            [
                self.weight[:3],
                self.weight[2:3],
                self.weight[3:],
                self.weight[-1:],
                self.weight.new_ones(1),
            ]
        )
>>>>>>> REPLACE