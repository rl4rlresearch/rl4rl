MECHANISM: Positional-factor scaling gauge fixing

HYPOTHESIS: Fixing `pos_proj[3, 0]` at its initialized value will reduce the model from 1,580 to 1,579 learned parameters while retaining at least 99% accuracy, because scaling positional-code column 3 and inversely scaling projection row 3 leaves their product unchanged.

INTENDED_EDIT: Store the initialized positional pivot as a fixed buffer, omit it from the learned flattened projection, and reconstruct the fourth projection row during every forward pass.

EVIDENCE: Six successive initialization-preserving positional-factor gauge reductions retained at least 99.94% accuracy, including 99.98% at 1,580 parameters; the rank-four factorization still has an exact per-latent-dimension scaling redundancy after exhausting its six orthogonal rotations.

<<<<<<< SEARCH
        self.pos_code = nn.Parameter(pos_code)
        self.pos_proj = nn.Parameter(
            torch.cat(
                (
                    pos_proj[0, 3:],
                    pos_proj[1, 2:],
                    pos_proj[2, 1:],
                    pos_proj[3],
                )
            )
        )
        self.pos_emb = None
=======
        self.pos_code = nn.Parameter(pos_code)
        # Fix one remaining scale-gauge coordinate at its initialized value.
        # Any change to this pivot is representable by inversely rescaling the
        # corresponding positional-code column and projection row.
        self.register_buffer("pos_proj_pivot", pos_proj[3, :1].clone())
        self.pos_proj = nn.Parameter(
            torch.cat(
                (
                    pos_proj[0, 3:],
                    pos_proj[1, 2:],
                    pos_proj[2, 1:],
                    pos_proj[3, 1:],
                )
            )
        )
        self.pos_emb = None
>>>>>>> REPLACE

<<<<<<< SEARCH
        fourth_proj_row = self.pos_proj[
            3 * self.cfg.d_model - 6 :
        ]
=======
        fourth_proj_row = torch.cat(
            (
                self.pos_proj_pivot,
                self.pos_proj[3 * self.cfg.d_model - 6 :],
            )
        )
>>>>>>> REPLACE