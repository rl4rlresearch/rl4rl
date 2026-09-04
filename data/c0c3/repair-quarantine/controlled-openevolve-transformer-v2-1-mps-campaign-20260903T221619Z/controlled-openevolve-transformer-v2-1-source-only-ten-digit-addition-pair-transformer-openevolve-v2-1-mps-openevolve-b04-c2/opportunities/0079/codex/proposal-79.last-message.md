MECHANISM: Seventh-column attention residual-output common-mode quotient

HYPOTHESIS: Quotienting attention projection columns 0, 1, 2, 4, 5, 6, and 7 while leaving optimization-sensitive column 3 unrestricted will produce a 1,587-parameter model with at least 99% accuracy.

INTENDED_EDIT: Extend the qualified alternate-column attention quotient to column 7, reconstruct the original column ordering, and retain full-coordinate AdamW optimization.

EVIDENCE: The 1,588-parameter reference achieved 99.36% accuracy with attention columns 0, 1, 2, 4, 5, and 6 compacted; quotienting adjacent column 7 is the closest incremental reduction that avoids the documented column-3 collapse.

<<<<<<< SEARCH
            # Uniform output-row components add only per-token common-mode
            # residual shifts, which subsequent LayerNorms erase. Quotient
            # the first two projection columns by that direction.
            compact_proj_columns = (
                self.proj_bias_basis.transpose(0, 1) @ proj_weight[:, :2]
            )
            remaining_proj_weight = proj_weight[:, 2:].clone()
=======
            # Uniform output-row components add only per-token common-mode
            # residual shifts, which subsequent LayerNorms erase. Quotient
            # every projection column except optimization-sensitive column 3.
            compact_proj_columns = (
                self.proj_bias_basis.transpose(0, 1)
                @ proj_weight[:, (0, 1, 2, 4, 5, 6, 7)]
            )
            remaining_proj_weight = proj_weight[:, 3:4].clone()
>>>>>>> REPLACE

<<<<<<< SEARCH
        proj_weight = torch.cat((compact_columns, self.proj.weight), dim=1)
=======
        proj_weight = torch.cat(
            (
                compact_columns[:, :3],
                self.proj.weight,
                compact_columns[:, 3:],
            ),
            dim=1,
        )
>>>>>>> REPLACE