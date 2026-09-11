MECHANISM: Learned adjacent final-normalization scale sharing

HYPOTHESIS: Sharing the final LayerNorm’s last two learned scale coordinates will reduce the model from 1,040 to 1,039 parameters while retaining at least 99% accuracy, because both coordinates remain adaptively scaled while all load-bearing normalization offsets remain independent.

INTENDED_EDIT: Store six final LayerNorm scale coordinates and reconstruct the seventh from its adjacent predecessor; retain the eighth coordinate as the fixed residual-scale reference.

EVIDENCE: Adjacent sharing of a final LayerNorm bias failed at 82.53%, showing that coordinate-specific offsets are load-bearing, while positional and scalar-MLP-bias ablations also failed; this motivates testing the still-uncompressed normalization-scale pathway with an adaptive tie rather than deleting or fixing a coordinate.

<<<<<<< SEARCH
        self.weight = nn.Parameter(torch.ones(normalized_shape - 1))
        self.bias = nn.Parameter(torch.zeros(normalized_shape - 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.layer_norm(
            x,
            self.normalized_shape,
            F.pad(self.weight, (0, 1), value=1.0),
            F.pad(self.bias, (0, 1)),
            self.eps,
        )
=======
        self.weight = nn.Parameter(torch.ones(normalized_shape - 2))
        self.bias = nn.Parameter(torch.zeros(normalized_shape - 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = torch.cat(
            [
                self.weight,
                self.weight[-1:],
                self.weight.new_ones(1),
            ]
        )
        return F.layer_norm(
            x,
            self.normalized_shape,
            weight,
            F.pad(self.bias, (0, 1)),
            self.eps,
        )
>>>>>>> REPLACE