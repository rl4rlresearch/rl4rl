MECHANISM: Second positional-factor scaling gauge fixing

HYPOTHESIS: Fixing `pos_proj[2, 1]` at initialization will reduce learned parameters from 1,579 to 1,578 while retaining at least 99% accuracy, because inverse scaling of positional-code column 2 preserves the represented positional table.

INTENDED_EDIT: Store the row-2 projection pivot as a fixed buffer, omit it from the learned flattened projection, and reconstruct it during every forward pass.

EVIDENCE: Fixing the analogous row-3 pivot achieved 99.87% accuracy with 1,579 parameters, demonstrating that an initialization-preserving positional scaling gauge can remove a learned scalar without losing required accuracy.

<<<<<<< SEARCH
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
=======
        # Fix two scale-gauge coordinates at their initialized values. Changes
        # to either pivot are representable by inversely rescaling the matching
        # positional-code column and projection row.
        self.register_buffer("pos_proj_row2_pivot", pos_proj[2, 1:2].clone())
        self.register_buffer("pos_proj_pivot", pos_proj[3, :1].clone())
        self.pos_proj = nn.Parameter(
            torch.cat(
                (
                    pos_proj[0, 3:],
                    pos_proj[1, 2:],
                    pos_proj[2, 2:],
                    pos_proj[3, 1:],
                )
            )
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
        third_proj_row = torch.cat(
            (
                self.pos_proj.new_zeros(1),
                self.pos_proj[
                    2 * self.cfg.d_model - 5 : 3 * self.cfg.d_model - 6
                ],
            )
        )
        fourth_proj_row = torch.cat(
            (
                self.pos_proj_pivot,
                self.pos_proj[3 * self.cfg.d_model - 6 :],
            )
        )
=======
        third_proj_row = torch.cat(
            (
                self.pos_proj.new_zeros(1),
                self.pos_proj_row2_pivot,
                self.pos_proj[
                    2 * self.cfg.d_model - 5 : 3 * self.cfg.d_model - 7
                ],
            )
        )
        fourth_proj_row = torch.cat(
            (
                self.pos_proj_pivot,
                self.pos_proj[3 * self.cfg.d_model - 7 :],
            )
        )
>>>>>>> REPLACE