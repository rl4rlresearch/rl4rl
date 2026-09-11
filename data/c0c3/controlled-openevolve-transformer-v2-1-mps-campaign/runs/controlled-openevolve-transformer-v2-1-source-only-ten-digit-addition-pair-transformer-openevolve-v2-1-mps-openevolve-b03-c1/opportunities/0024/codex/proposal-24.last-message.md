MECHANISM: Third orthogonal gauge fixing of the positional factors

HYPOTHESIS: Removing a third rotationally redundant scalar from the rank-four positional factors will reduce parameters from 1,584 to 1,583 while retaining at least 99% accuracy, because another orthogonal latent-basis rotation preserves the initialized positional table and factor geometry exactly.

INTENDED_EDIT: Rotate positional latent dimensions 2 and 3 so `pos_proj[2, 0]` is zero, omit that scalar, and reconstruct all three fixed zeros during the forward pass.

EVIDENCE: The first and second initialization-preserving positional-factor gauge reductions both retained 99.95% or better accuracy while reducing one parameter each, directly motivating one more incremental application of the same invariance.

<<<<<<< SEARCH
        # then replace it with its best rank-four factorization. Orthogonal
        # latent-basis rotations fix two redundant projection entries at zero
        # without changing the initialized positional table or factor geometry.
=======
        # then replace it with its best rank-four factorization. Orthogonal
        # latent-basis rotations fix three redundant projection entries at zero
        # without changing the initialized positional table or factor geometry.
>>>>>>> REPLACE

<<<<<<< SEARCH
            a = pos_proj[1, 0]
            b = pos_proj[2, 0]
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
                    pos_proj[0, 1:],
                    pos_proj[1, 1:],
                    pos_proj[2:].reshape(-1),
                )
            )
        )
=======
            a = pos_proj[1, 0]
            b = pos_proj[2, 0]
            radius = torch.sqrt(a.square() + b.square())
            rotation = torch.eye(4, device=pos_proj.device, dtype=pos_proj.dtype)
            rotation[1, 1] = b / radius
            rotation[1, 2] = -a / radius
            rotation[2, 1] = a / radius
            rotation[2, 2] = b / radius
            pos_code = pos_code @ rotation.T
            pos_proj = rotation @ pos_proj

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
>>>>>>> REPLACE

<<<<<<< SEARCH
        pos_proj = torch.cat(
            (
                first_proj_row.unsqueeze(0),
                second_proj_row.unsqueeze(0),
                self.pos_proj[2 * (self.cfg.d_model - 1) :].view(
                    2, self.cfg.d_model
                ),
            ),
            dim=0,
        )
=======
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
        pos_proj = torch.cat(
            (
                first_proj_row.unsqueeze(0),
                second_proj_row.unsqueeze(0),
                third_proj_row.unsqueeze(0),
                fourth_proj_row.unsqueeze(0),
            ),
            dim=0,
        )
>>>>>>> REPLACE