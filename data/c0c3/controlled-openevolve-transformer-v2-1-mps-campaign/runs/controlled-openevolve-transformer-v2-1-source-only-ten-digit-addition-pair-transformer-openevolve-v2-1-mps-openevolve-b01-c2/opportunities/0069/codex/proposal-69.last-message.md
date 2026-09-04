MECHANISM: Orthonormal final-MLP common-bias gauge

HYPOTHESIS: Reparameterizing the final MLP output bias in an orthonormal zero-sum basis will reduce the verified model from 1,573 to 1,572 parameters while maintaining at least 99% accuracy.

INTENDED_EDIT: Replace the MLP’s `fc2` with the existing orthonormal common-bias-gauged linear layer, removing the bias direction erased by the final LayerNorm.

EVIDENCE: The orthonormal attention-projection bias gauge achieved 99.97% at 1,573 parameters, while the coordinate-anchored final-MLP bias gauge collapsed to 12.32%; this tests the same final-LayerNorm null direction using the successful symmetric, isometric chart.

<<<<<<< SEARCH
        self.fc2 = nn.Linear(d_ff, d_model)
=======
        self.fc2 = OrthonormalCommonBiasGaugedLinear(d_ff, d_model)
>>>>>>> REPLACE