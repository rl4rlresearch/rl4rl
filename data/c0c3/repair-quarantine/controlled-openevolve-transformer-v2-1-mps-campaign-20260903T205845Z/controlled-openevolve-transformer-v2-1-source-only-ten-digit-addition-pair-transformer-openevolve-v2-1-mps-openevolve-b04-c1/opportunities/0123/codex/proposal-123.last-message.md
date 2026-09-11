MECHANISM: Single-coordinate final-normalization scale anchor

HYPOTHESIS: Fixing one zero-deviation final LayerNorm scale at its initialization value of 1 will reduce the verified model from 847 to 846 parameters while retaining at least 99% accuracy.

INTENDED_EDIT: Learn three of the four readout-visible LayerNorm scales and materialize the fourth as a fixed unit scale.

EVIDENCE: The 847-parameter model reached 99.94% accuracy, while constraints on independent GLU projections caused large accuracy losses; this isolates a single readout-scale degree of freedom without changing GLU or attention capacity.

<<<<<<< SEARCH
        self.weight = nn.Parameter(torch.ones(output_dim))
        self.bias = nn.Parameter(torch.zeros(output_dim))
=======
        self.weight = nn.Parameter(torch.ones(output_dim - 1))
        self.bias = nn.Parameter(torch.zeros(output_dim))
>>>>>>> REPLACE

<<<<<<< SEARCH
        tail = self.normalized_shape[0] - self.output_dim
        weight = F.pad(self.weight, (0, tail), value=1.0)
        bias = F.pad(self.bias, (0, tail))
=======
        tail = self.normalized_shape[0] - self.output_dim
        weight = F.pad(self.weight, (0, tail + 1), value=1.0)
        bias = F.pad(self.bias, (0, tail))
>>>>>>> REPLACE