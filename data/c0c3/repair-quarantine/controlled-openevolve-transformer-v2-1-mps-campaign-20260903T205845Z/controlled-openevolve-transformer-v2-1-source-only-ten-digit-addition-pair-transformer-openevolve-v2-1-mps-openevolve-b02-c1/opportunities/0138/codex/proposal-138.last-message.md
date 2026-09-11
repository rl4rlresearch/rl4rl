MECHANISM: Chained width-one MLP coordinate sharing

HYPOTHESIS: Sharing residual coordinate six’s MLP input coefficient with coordinates four and five will reduce the verified model from 984 to 983 parameters while retaining at least 99% accuracy, because coordinate six already shares its final-LayerNorm scale with coordinate five.

INTENDED_EDIT: Store three MLP input coefficients instead of four, reconstructing the seven-coordinate input pattern as three groups: 0–1, 2–3, and 4–6, while preserving the original initialization RNG stream.

EVIDENCE: The current 984-parameter model achieved 99.75% accuracy with successful coordinate-aligned sharing, whereas reductions to final-LayerNorm scales, relative bias, and companion keys failed; this tests an orthogonal reduction in the width-one MLP along an existing final-normalization pairing.

<<<<<<< SEARCH
        self.fc1 = nn.Linear(d_model - 4, d_ff, bias=False)
        self.fc1._removed_input_features = 3

        # Preserve the constructor RNG stream of the removed input weights.
        fc1_weight_bound = 1.0 / math.sqrt(d_model - 1)
        torch.empty(3 * d_ff).uniform_(
            -fc1_weight_bound, fc1_weight_bound
        )
=======
        self.fc1 = nn.Linear(d_model - 5, d_ff, bias=False)
        self.fc1._removed_input_features = 4

        # Preserve the constructor RNG stream of the removed input weights.
        fc1_weight_bound = 1.0 / math.sqrt(d_model - 1)
        torch.empty(4 * d_ff).uniform_(
            -fc1_weight_bound, fc1_weight_bound
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
                self.fc1.weight[:, 2:3],
                self.fc1.weight[:, 2:3],
                self.fc1.weight[:, 3:],
=======
                self.fc1.weight[:, 2:3],
                self.fc1.weight[:, 2:3],
                self.fc1.weight[:, 2:3],
>>>>>>> REPLACE