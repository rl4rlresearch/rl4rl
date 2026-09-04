MECHANISM: Third-column attention residual-output common-mode quotient

HYPOTHESIS: Extending the verified 1,592-parameter design by quotienting the third attention output-projection column will produce a 1,591-parameter model with at least 99% accuracy, because its uniform residual-channel component is erased by subsequent LayerNorms.

INTENDED_EDIT: Reproduce the qualified all-but-column-3 `fc2` compaction, then parameterize the first three attention projection columns in the seven-dimensional zero-sum basis; the existing matrix-aware optimizer continues to maintain full-coordinate AdamW moments.

EVIDENCE: Reference Design 2 achieved 99.97% accuracy at 1,592 parameters with every `fc2` column except column 3 compacted, while the qualified architecture already compacts the first two attention projection columns through the same exact LayerNorm-null direction; extending that adjacent attention quotient is the closest untested reduction.

<<<<<<< SEARCH
        # each coefficient made zero. Then remove the function-null uniform
        # output component from the first two projection columns.
=======
        # each coefficient made zero. Then remove the function-null uniform
        # output component from the first three projection columns.
>>>>>>> REPLACE

<<<<<<< SEARCH
            compact_proj_columns = (
                self.proj_bias_basis.transpose(0, 1) @ proj_weight[:, :2]
            )
            remaining_proj_weight = proj_weight[:, 2:].clone()
=======
            compact_proj_columns = (
                self.proj_bias_basis.transpose(0, 1) @ proj_weight[:, :3]
            )
            remaining_proj_weight = proj_weight[:, 3:].clone()
>>>>>>> REPLACE

<<<<<<< SEARCH
        # Uniform output components are erased by the final LayerNorm.
        # Quotient columns zero through two plus columns four through nine,
        # retaining column three as an unrestricted optimization coordinate.
        with torch.no_grad():
            weight = self.fc2.weight.detach()
            compact_columns = (
                self.bias_basis.transpose(0, 1)
                @ weight[:, (0, 1, 2, 4, 5, 6, 7, 8, 9)]
            )
            remaining_weight = torch.cat(
                (weight[:, 3:4], weight[:, 10:]), dim=1
            ).clone()
=======
        # Uniform output components are erased by the final LayerNorm.
        # Quotient every column except column three, which remains an
        # unrestricted optimization coordinate.
        with torch.no_grad():
            weight = self.fc2.weight.detach()
            compact_columns = (
                self.bias_basis.transpose(0, 1)
                @ weight[:, (0, 1, 2, 4, 5, 6, 7, 8, 9, 10, 11)]
            )
            remaining_weight = weight[:, 3:4].clone()
>>>>>>> REPLACE

<<<<<<< SEARCH
        weight = torch.cat(
            (
                compact_columns[:, :3],
                self.fc2.weight[:, :1],
                compact_columns[:, 3:],
                self.fc2.weight[:, 1:],
            ),
            dim=1,
        )
=======
        weight = torch.cat(
            (
                compact_columns[:, :3],
                self.fc2.weight,
                compact_columns[:, 3:],
            ),
            dim=1,
        )
>>>>>>> REPLACE