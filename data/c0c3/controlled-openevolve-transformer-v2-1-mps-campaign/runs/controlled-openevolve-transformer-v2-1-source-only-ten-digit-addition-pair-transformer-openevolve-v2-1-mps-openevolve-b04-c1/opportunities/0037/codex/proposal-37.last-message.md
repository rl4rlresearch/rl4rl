MECHANISM: Third MLP hidden-channel shift quotient

HYPOTHESIS: Constraining a third `fc2` weight column to the seven-dimensional zero-mean subspace will reduce the model from 1603 to 1602 parameters while retaining at least 99% accuracy, because its discarded all-ones component produces only an activation-dependent residual shift erased by the final LayerNorm.

INTENDED_EDIT: Extend `OneColumnGaugeLinear` from two to three independently parameterized zero-mean output columns, leaving the other nine `fc2` columns dense.

EVIDENCE: Applying the same quotient to two `fc2` columns achieved 99.98% accuracy with 1603 parameters, making a third exchangeable hidden channel the most direct evidence-backed reduction.

<<<<<<< SEARCH
        self.rest = nn.Linear(in_features - 2, out_features, bias=False)
        self.first_coordinates = nn.Parameter(torch.empty(2, out_features - 1))
=======
        self.rest = nn.Linear(in_features - 3, out_features, bias=False)
        self.first_coordinates = nn.Parameter(torch.empty(3, out_features - 1))
>>>>>>> REPLACE