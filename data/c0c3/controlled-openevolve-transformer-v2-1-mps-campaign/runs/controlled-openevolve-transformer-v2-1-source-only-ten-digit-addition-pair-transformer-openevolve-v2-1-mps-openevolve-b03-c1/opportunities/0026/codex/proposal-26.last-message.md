MECHANISM: Fifth orthogonal gauge fixing of rank-four positional factors

HYPOTHESIS: Rotating latent dimensions 1 and 2 to fix `pos_proj[1, 1]` at zero will reduce the model from 1,582 to 1,581 parameters while retaining at least 99% accuracy, because this orthogonal rotation preserves the initialized positional table and all four existing zero constraints exactly.

INTENDED_EDIT: Apply an initialization-preserving rotation between positional latent dimensions 1 and 2, omit the newly fixed projection scalar, and reconstruct the five fixed zeros during each forward pass.

EVIDENCE: Four successive orthogonal positional-factor gauge reductions retained at least 99.94% accuracy, including the latest 1,582-parameter model; dimensions 1 and 2 both remain zero in the first projection column, so their residual rotation can eliminate a second-column scalar without disturbing prior constraints.

<<<<<<< SEARCH
            pos_code = pos_code @ rotation.T
            pos_proj = rotation @ pos_proj

        self.pos_code = nn.Parameter(pos_code)
=======
            pos_code = pos_code @ rotation.T
            pos_proj = rotation @ pos_proj

            # Rows one and two remain zero in the first column, so rotate them
            # to eliminate another second-column scalar without disturbing any
            # of the four previously fixed entries.
            a = pos_proj[1, 1]
            b = pos_proj[2, 1]
            radius = torch.sqrt(a.square() + b.square())
            rotation = torch.eye(4, device=pos_proj.device, dtype=pos_proj.dtype)
            rotation[1, 1] = b / radius
            rotation[1, 2] = -a / radius
            rotation[2, 1] = a / radius
            rotation[2, 2] = b / radius
            pos_code = pos_code @ rotation.T
            pos_proj = rotation @ pos_proj

        self.pos_code = nn.Parameter(pos_code)
>>>>>>> REPLACE

<<<<<<< SEARCH
                    pos_proj[0, 2:],
                    pos_proj[1, 1:],
                    pos_proj[2, 1:],
                    pos_proj[3],
=======
                    pos_proj[0, 2:],
                    pos_proj[1, 2:],
                    pos_proj[2, 1:],
                    pos_proj[3],
>>>>>>> REPLACE

<<<<<<< SEARCH
        second_proj_row = torch.cat(
            (
                self.pos_proj.new_zeros(1),
                self.pos_proj[
                    self.cfg.d_model - 2 : 2 * self.cfg.d_model - 3
                ],
            )
        )
        third_proj_row = torch.cat(
            (
                self.pos_proj.new_zeros(1),
                self.pos_proj[
                    2 * self.cfg.d_model - 3 : 3 * self.cfg.d_model - 4
                ],
            )
        )
        fourth_proj_row = self.pos_proj[
            3 * self.cfg.d_model - 4 :
        ]
=======
        second_proj_row = torch.cat(
            (
                self.pos_proj.new_zeros(2),
                self.pos_proj[
                    self.cfg.d_model - 2 : 2 * self.cfg.d_model - 4
                ],
            )
        )
        third_proj_row = torch.cat(
            (
                self.pos_proj.new_zeros(1),
                self.pos_proj[
                    2 * self.cfg.d_model - 4 : 3 * self.cfg.d_model - 5
                ],
            )
        )
        fourth_proj_row = self.pos_proj[
            3 * self.cfg.d_model - 5 :
        ]
>>>>>>> REPLACE