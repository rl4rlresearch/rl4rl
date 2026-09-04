MECHANISM: Sixth orthogonal gauge fixing of rank-four positional factors

HYPOTHESIS: Rotating latent dimensions 0 and 1 to fix `pos_proj[0, 2]` at zero will reduce the model from 1,581 to 1,580 parameters while retaining at least 99% accuracy, because both rows are already zero in the first two projection columns, so this rotation preserves the initialized positional table and all five existing constraints exactly.

INTENDED_EDIT: Apply an initialization-preserving rotation between positional latent dimensions 0 and 1, omit the newly fixed scalar, and reconstruct the six fixed zeros during every forward pass.

EVIDENCE: Five successive orthogonal positional-factor gauge reductions retained at least 99.94% accuracy, with the latest 1,581-parameter design reaching 99.98%; the remaining rotation between rows 0 and 1 can eliminate a third-column scalar without disturbing prior constraints.

<<<<<<< SEARCH
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
        self.pos_proj = nn.Parameter(
            torch.cat(
                (
                    pos_proj[0, 2:],
                    pos_proj[1, 2:],
                    pos_proj[2, 1:],
                    pos_proj[3],
                )
            )
        )
=======
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

            # Rows zero and one are both zero in the first two columns, leaving
            # one residual rotation that can eliminate a third-column scalar.
            a = pos_proj[0, 2]
            b = pos_proj[1, 2]
            radius = torch.sqrt(a.square() + b.square())
            rotation = torch.eye(4, device=pos_proj.device, dtype=pos_proj.dtype)
            rotation[0, 0] = b / radius
            rotation[0, 1] = -a / radius
            rotation[1, 0] = a / radius
            rotation[1, 1] = b / radius
            pos_code = pos_code @ rotation.T
            pos_proj = rotation @ pos_proj

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
>>>>>>> REPLACE

<<<<<<< SEARCH
        first_proj_row = torch.cat(
            (self.pos_proj.new_zeros(2), self.pos_proj[: self.cfg.d_model - 2])
        )
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
=======
        first_proj_row = torch.cat(
            (self.pos_proj.new_zeros(3), self.pos_proj[: self.cfg.d_model - 3])
        )
        second_proj_row = torch.cat(
            (
                self.pos_proj.new_zeros(2),
                self.pos_proj[
                    self.cfg.d_model - 3 : 2 * self.cfg.d_model - 5
                ],
            )
        )
        third_proj_row = torch.cat(
            (
                self.pos_proj.new_zeros(1),
                self.pos_proj[
                    2 * self.cfg.d_model - 5 : 3 * self.cfg.d_model - 6
                ],
            )
        )
        fourth_proj_row = self.pos_proj[
            3 * self.cfg.d_model - 6 :
        ]
>>>>>>> REPLACE