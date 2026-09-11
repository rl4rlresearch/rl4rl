MECHANISM: Fixed-spectrum positional clock

HYPOTHESIS: Constraining the four harmonic amplitudes to their initialized ratios while learning one global amplitude will reduce the verified model from 606 to 603 parameters and retain at least 99% accuracy, because the learned query/key maps can still mix each positional axis independently and the initialized function is preserved.

INTENDED_EDIT: Factor the four-parameter positional scale vector into a fixed normalized spectral shape and one learned global scale.

EVIDENCE: The 606-parameter model reached 99.89% accuracy using fixed harmonic codes and fixed orthogonal positional directions with fully learned query/key maps. The catastrophic non-affine final-LayerNorm result shows lexical readout geometry is load-bearing, motivating compression of the more structurally redundant positional amplitudes instead.

<<<<<<< SEARCH
        # Give each harmonic a fixed orthogonal direction in the observable
        # zero-mean residual subspace. Learned query/key maps can perform the
        # required mixing, so retain only one trainable amplitude per harmonic.
=======
        # Give each harmonic a fixed orthogonal direction in the observable
        # zero-mean residual subspace. Preserve the initialized spectral shape,
        # while learning one global position strength; query/key maps retain
        # independent learned mixing of the four harmonic axes.
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.pos_scale = nn.Parameter(
            torch.linalg.vector_norm(initialized_pos_rows, dim=1)
        )
        self.pos_emb = None
=======
        initialized_pos_scale = torch.linalg.vector_norm(
            initialized_pos_rows, dim=1
        )
        initialized_pos_amplitude = torch.linalg.vector_norm(
            initialized_pos_scale
        )
        self.register_buffer(
            "pos_scale_shape",
            initialized_pos_scale / initialized_pos_amplitude,
        )
        self.pos_scale = nn.Parameter(initialized_pos_amplitude.clone())
        self.pos_emb = None
>>>>>>> REPLACE

<<<<<<< SEARCH
        pos_proj = self.pos_scale.unsqueeze(1) * self.pos_basis
=======
        pos_scale = self.pos_scale * self.pos_scale_shape
        pos_proj = pos_scale.unsqueeze(1) * self.pos_basis
>>>>>>> REPLACE