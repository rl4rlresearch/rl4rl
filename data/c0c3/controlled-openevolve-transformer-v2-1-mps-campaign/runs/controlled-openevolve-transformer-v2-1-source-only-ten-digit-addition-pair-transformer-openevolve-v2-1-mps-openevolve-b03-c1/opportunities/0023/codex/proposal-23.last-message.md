MECHANISM: Second orthogonal gauge fixing of the positional factors

HYPOTHESIS: Removing a second rotationally redundant scalar from the rank-four positional factors will reduce parameters from 1,585 to 1,584 while retaining at least 99% accuracy, because a second orthogonal basis rotation preserves both the initialized positional table and latent-factor geometry.

INTENDED_EDIT: Rotate positional latent dimensions 1 and 2 so `pos_proj[1, 0]` is zero, omit that scalar, and reconstruct both fixed zeros during the forward pass.

EVIDENCE: The first initialization-preserving positional-factor gauge reduction achieved 99.96% accuracy with 1,585 parameters, directly motivating another one-scalar removal using the same exact invariance.

<<<<<<< SEARCH
        # Preserve the baseline RNG stream by initializing the original table,
        # then replace it with its best rank-four factorization. An orthogonal
        # latent-basis rotation fixes one redundant projection entry at zero
        # without changing the initialized positional table.
        with torch.no_grad():
            left, singular, right = torch.linalg.svd(
                self.pos_emb.weight, full_matrices=False
            )
            scale = singular[:4].sqrt()
            pos_code = left[:, :4] * scale
            pos_proj = scale.unsqueeze(1) * right[:4]

            a = pos_proj[0, 0]
            b = pos_proj[1, 0]
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
            torch.cat((pos_proj[0, 1:], pos_proj[1:].reshape(-1)))
        )
=======
        # Preserve the baseline RNG stream by initializing the original table,
        # then replace it with its best rank-four factorization. Orthogonal
        # latent-basis rotations fix two redundant projection entries at zero
        # without changing the initialized positional table or factor geometry.
        with torch.no_grad():
            left, singular, right = torch.linalg.svd(
                self.pos_emb.weight, full_matrices=False
            )
            scale = singular[:4].sqrt()
            pos_code = left[:, :4] * scale
            pos_proj = scale.unsqueeze(1) * right[:4]

            a = pos_proj[0, 0]
            b = pos_proj[1, 0]
            radius = torch.sqrt(a.square() + b.square())
            rotation = torch.eye(4, device=pos_proj.device, dtype=pos_proj.dtype)
            rotation[0, 0] = b / radius
            rotation[0, 1] = -a / radius
            rotation[1, 0] = a / radius
            rotation[1, 1] = b / radius
            pos_code = pos_code @ rotation.T
            pos_proj = rotation @ pos_proj

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
>>>>>>> REPLACE

<<<<<<< SEARCH
        first_proj_row = torch.cat(
            (self.pos_proj.new_zeros(1), self.pos_proj[: self.cfg.d_model - 1])
        )
        pos_proj = torch.cat(
            (
                first_proj_row.unsqueeze(0),
                self.pos_proj[self.cfg.d_model - 1 :].view(3, self.cfg.d_model),
            ),
            dim=0,
        )
=======
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
>>>>>>> REPLACE