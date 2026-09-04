MECHANISM: Sixth-column attention residual-output common-mode quotient

HYPOTHESIS: Quotienting attention projection columns 0, 1, 2, 4, 5, and 6 while leaving optimization-sensitive column 3 unrestricted will produce a 1,588-parameter model with at least 99% accuracy.

INTENDED_EDIT: Extend the qualified alternate-column attention quotient through column 6, preserve original column ordering during reconstruction, and retain full-coordinate AdamW optimization.

EVIDENCE: Reference Design 3 achieved 99.91% accuracy at 1,589 parameters with attention columns 0, 1, 2, 4, and 5 compacted; adding adjacent column 6 is the closest incremental reduction that avoids the documented column-3 collapse.

<<<<<<< SEARCH
        # Rotate value channels, counter-rotate their output columns, and omit
        # each coefficient made zero. Then remove the function-null uniform
        # output component from the first three projection columns.
=======
        # Rotate value channels, counter-rotate their output columns, and omit
        # each coefficient made zero. Also quotient projection columns zero
        # through two and four through six, leaving column three unrestricted.
>>>>>>> REPLACE

<<<<<<< SEARCH
            compact_proj_columns = (
                self.proj_bias_basis.transpose(0, 1) @ proj_weight[:, :3]
            )
            remaining_proj_weight = proj_weight[:, 3:].clone()
=======
            compact_proj_columns = (
                self.proj_bias_basis.transpose(0, 1)
                @ proj_weight[:, (0, 1, 2, 4, 5, 6)]
            )
            remaining_proj_weight = torch.cat(
                (proj_weight[:, 3:4], proj_weight[:, 7:]), dim=1
            ).clone()
>>>>>>> REPLACE

<<<<<<< SEARCH
        proj_weight = torch.cat((compact_columns, self.proj.weight), dim=1)
=======
        proj_weight = torch.cat(
            (
                compact_columns[:, :3],
                self.proj.weight[:, :1],
                compact_columns[:, 3:],
                self.proj.weight[:, 1:],
            ),
            dim=1,
        )
>>>>>>> REPLACE