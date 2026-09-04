MECHANISM: Shared-gain fixed-spectrum positional encoding

HYPOTHESIS: Replacing four independently learned harmonic amplitudes with their exact initialized spectral template plus one learned global gain will reduce the verified model from 606 to 603 parameters while retaining at least 99% accuracy.

INTENDED_EDIT: Preserve the initial positional representation exactly, freeze its relative harmonic amplitudes, and learn only a shared additive positional-strength adjustment.

EVIDENCE: The 606-parameter model reached 0.9989 accuracy using fixed harmonic position codes and learned relative-distance routing, while the failed 604-parameter experiment altered the fragile lexical representation; this instead tests whether independent positional amplitudes are necessary while leaving lexical computation unchanged.

<<<<<<< SEARCH
        # Give each harmonic a fixed orthogonal direction in the observable
        # zero-mean residual subspace. Learned query/key maps can perform the
        # required mixing, so retain only one trainable amplitude per harmonic.
=======
        # Give each harmonic a fixed orthogonal direction in the observable
        # zero-mean residual subspace. Preserve the initialized spectral shape
        # as a checkpointed buffer and learn only their shared global gain.
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
        self.register_buffer("pos_scale_base", initialized_pos_scale)
        self.pos_scale = nn.Parameter(
            initialized_pos_scale.new_zeros(1)
        )
        self.pos_emb = None
>>>>>>> REPLACE

<<<<<<< SEARCH
        pos_proj = self.pos_scale.unsqueeze(1) * self.pos_basis
=======
        pos_scale = self.pos_scale_base + self.pos_scale
        pos_proj = pos_scale.unsqueeze(1) * self.pos_basis
>>>>>>> REPLACE