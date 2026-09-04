MECHANISM: Final-LayerNorm quotient for attention output bias

HYPOTHESIS: Representing the attention output-projection bias with seven mean-free coordinates will reduce parameters from 1,617 to 1,616 while retaining at least 99% accuracy, because its omitted all-ones component propagates unchanged through the residual stream and is canceled by downstream LayerNorms.

INTENDED_EDIT: Reuse `MeanFreeResidualLinear` for the attention output projection, preserving its full weight matrix, all seven observable bias directions, and constructor RNG consumption.

EVIDENCE: The identical mean-free quotient applied to the MLP output bias reduced the model from 1,618 to 1,617 parameters while retaining 99.98% accuracy; the attention output bias has the same downstream constant-shift invariance.

<<<<<<< SEARCH
        self.proj = nn.Linear(d_model, d_model)
=======
        self.proj = MeanFreeResidualLinear(d_model, d_model)
>>>>>>> REPLACE