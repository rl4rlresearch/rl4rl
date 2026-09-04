MECHANISM: Early-pair final-LayerNorm scale sharing

HYPOTHESIS: Sharing final-LayerNorm scales for residual coordinates zero and one will reduce the model from 984 to 983 parameters while retaining at least 99% accuracy, because those coordinates already share an MLP input coefficient, while all scale and bias distinctions implicated by prior failures remain unchanged.

INTENDED_EDIT: Store four final-LayerNorm scales instead of five and reuse the first scale for coordinates zero and one, preserving the existing sharing for coordinates two/three and five/six.

EVIDENCE: Mid-pair final-LayerNorm scale sharing achieved 99.75% at 984 parameters when aligned with an existing MLP input pair; applying the same isolated constraint to the other early MLP-shared pair is the smallest informative continuation, unlike the failed extension of the sensitive terminal scale group.

<<<<<<< SEARCH
        self.weight = nn.Parameter(torch.ones(normalized_shape - 3))
        self.bias = nn.Parameter(torch.zeros(normalized_shape - 6))
=======
        self.weight = nn.Parameter(torch.ones(normalized_shape - 4))
        self.bias = nn.Parameter(torch.zeros(normalized_shape - 6))
>>>>>>> REPLACE

<<<<<<< SEARCH
        weight = torch.cat(
            [
                self.weight[:3],
                self.weight[2:3],
                self.weight[3:],
                self.weight[-1:],
                self.weight.new_ones(1),
            ]
        )
=======
        weight = torch.cat(
            [
                self.weight[:1],
                self.weight[:1],
                self.weight[1:2],
                self.weight[1:2],
                self.weight[2:],
                self.weight[-1:],
                self.weight.new_ones(1),
            ]
        )
>>>>>>> REPLACE