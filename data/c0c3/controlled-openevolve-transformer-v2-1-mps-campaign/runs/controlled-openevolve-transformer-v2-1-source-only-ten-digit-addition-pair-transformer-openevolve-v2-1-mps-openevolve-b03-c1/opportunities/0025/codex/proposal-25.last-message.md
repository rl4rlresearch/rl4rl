MECHANISM: Residual orthogonal gauge fixing of rank-four positional factors

HYPOTHESIS: Rotating latent dimensions 0 and 1 to fix `pos_proj[0, 1]` at zero will reduce the model from 1,583 to 1,582 parameters while retaining at least 99% accuracy, because the rotation preserves the initialized positional table and the three existing first-column constraints exactly.

INTENDED_EDIT: Apply a fourth initialization-preserving latent-basis rotation, omit the newly fixed projection scalar, and reconstruct the four fixed zeros during each forward pass.

EVIDENCE: Three successive orthogonal positional-factor gauge reductions retained 99.96%, 99.95%, and 99.98% accuracy; after aligning the first projection column, dimensions 0 and 1 retain an unused rotational degree of freedom that can eliminate one entry in the second column.

<<<<<<< SEARCH
            a = pos_proj[2, 0]
            b = pos_proj[3, 0]
            radius = torch.sqrt(a.square() + b.square())
            rotation = torch.eye(4, device=pos_proj.device, dtype=pos_proj.dtype)
            rotation[2, 2] = b / radius
            rotation[2, 3] = -a / radius
            rotation[3, 2] = a / radius
            rotation[3, 3] = b / radius
            pos_code = pos_code @ rotation.T
            pos_proj = rotation @ pos_proj

        self.pos_code = nn.Parameter(pos_code)
        self.pos_proj = nn.Parameter(
            torch.cat(
                (
                    pos_proj[0, 1:],
                    pos_proj[1, 1:],
                    pos_proj[2, 1:],
                    pos_proj[3],
                )
            )
        )
=======
            a = pos_proj[2, 0]
            b = pos_proj[3, 0]
            radius = torch.sqrt(a.square() + b.square())
            rotation = torch.eye(4, device=pos_proj.device, dtype=pos_proj.dtype)
            rotation[2, 2] = b / radius
            rotation[2, 3] = -a / radius
            rotation[3, 2] = a / radius
            rotation[3, 3] = b / radius
            pos_code = pos_code @ rotation.T
            pos_proj = rotation @ pos_proj

            # Rows zero and one already have zero first coordinates, so their
            # remaining rotation can eliminate another projection scalar while
            # preserving all three first-column constraints.
            a = pos_proj[0, 1]
            b = pos_proj[1, 1]
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
                    pos_proj[0, 2:],
                    pos_proj[1, 1:],
                    pos_proj[2, 1:],
                    pos_proj[3],
                )
            )
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
        first_proj_row = torch.cat(
            (self.pos_proj.new_zeros(1), self.pos_proj[: self.cfg.d_model - 1])
        )
        second_proj_row = torch.cat(
            (
                self.pos_proj.new_zeros(1),
                self.pos_proj[
                    self.cfg.d_model - 1 : 2 * (self.cfg.d_model - 1)
                ],
            )
        )
        third_proj_row = torch.cat(
            (
                self.pos_proj.new_zeros(1),
                self.pos_proj[
                    2 * (self.cfg.d_model - 1) : 3 * (self.cfg.d_model - 1)
                ],
            )
        )
        fourth_proj_row = self.pos_proj[
            3 * (self.cfg.d_model - 1) :
        ]
=======
        first_proj_row = torch.cat(
            (self.pos_proj.new_zeros(2), self.pos_proj[: self.cfg.d_model - 2])
        )
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
>>>>>>> REPLACE