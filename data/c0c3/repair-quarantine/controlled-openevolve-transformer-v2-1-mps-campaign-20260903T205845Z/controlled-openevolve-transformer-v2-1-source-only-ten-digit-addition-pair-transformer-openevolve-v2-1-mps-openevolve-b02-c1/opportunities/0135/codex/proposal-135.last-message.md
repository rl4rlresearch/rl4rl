MECHANISM: Pairwise final-normalization scale sharing

HYPOTHESIS: Tying final-LayerNorm scales for residual coordinates zero and one will reduce the model from 984 to 983 parameters while retaining at least 99% accuracy, because those coordinates already share one learned MLP input coefficient and the verified 984-parameter model has a 0.75-point accuracy margin.

INTENDED_EDIT: Store four final-LayerNorm scales and reconstruct scales as paired coordinates 0–1 and 2–3, independent coordinate 4, paired coordinates 5–6, and fixed coordinate 7.

EVIDENCE: The current 984-parameter design achieved 99.75% accuracy after sharing coordinates two and three, showing that structurally aligned final-LayerNorm scale sharing can remove a parameter without approaching the 99% threshold.

<<<<<<< SEARCH
        self.weight = nn.Parameter(torch.ones(normalized_shape - 3))
=======
        self.weight = nn.Parameter(torch.ones(normalized_shape - 4))
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
                self.weight[0:1],
                self.weight[0:1],
                self.weight[1:2],
                self.weight[1:2],
                self.weight[2:],
                self.weight[-1:],
                self.weight.new_ones(1),
            ]
        )
>>>>>>> REPLACE