MECHANISM: Zero-mean MLP output-bias gauge fixing

HYPOTHESIS: Compressing the MLP output bias from eight coordinates to the seven-dimensional zero-mean subspace will reduce the model from 1,120 to 1,119 learned parameters while retaining at least 99% accuracy, because any discarded all-ones component is exactly removed by downstream LayerNorms.

INTENDED_EDIT: Represent `fc2.bias` in the existing orthonormal zero-mean basis and reconstruct its full bias during the forward pass.

EVIDENCE: The 1,120-parameter design reached 99.65%, whereas removing the twelfth-farthest attention biases collapsed accuracy to 8.01%; this motivates preserving the load-bearing routing endpoint and instead eliminating an exact LayerNorm-invariant MLP bias degree of freedom.

<<<<<<< SEARCH
        self.register_buffer("fc1_basis", basis.T, persistent=False)
=======
        self.register_buffer("fc1_basis", basis.T, persistent=False)

        # An all-ones output-bias component propagates only as a residual
        # shift and is removed by every downstream LayerNorm. Retain exactly
        # the observable zero-mean component in the same orthonormal basis.
        self.fc2.bias = nn.Parameter(
            (self.fc1_basis @ self.fc2.bias).clone()
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
        fc2_weight = self.fc1_basis.T @ self.fc2.weight
        output = F.linear(F.gelu(hidden), fc2_weight, self.fc2.bias)
=======
        fc2_weight = self.fc1_basis.T @ self.fc2.weight
        fc2_bias = self.fc1_basis.T @ self.fc2.bias
        output = F.linear(F.gelu(hidden), fc2_weight, fc2_bias)
>>>>>>> REPLACE