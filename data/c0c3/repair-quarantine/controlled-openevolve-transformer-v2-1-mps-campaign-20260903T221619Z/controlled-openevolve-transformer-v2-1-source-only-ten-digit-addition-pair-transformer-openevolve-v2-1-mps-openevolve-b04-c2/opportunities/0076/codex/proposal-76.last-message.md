MECHANISM: Alternate-column attention residual-output quotient

HYPOTHESIS: Quotienting attention projection columns 0, 1, 2, and 4 while leaving optimization-sensitive column 3 unrestricted will produce a 1,590-parameter model with at least 99% accuracy.

INTENDED_EDIT: Reproduce the qualified all-but-column-3 MLP quotient, then compact attention projection columns 0, 1, 2, and 4 with full-coordinate AdamW moments and reconstruct their original ordering during the forward pass.

EVIDENCE: The 1,591-parameter design with attention columns 0–2 compacted achieved 99.81%, whereas adding column 3 collapsed to 68.17%; the analogous MLP collapse was avoided by leaving column 3 unrestricted and quotienting column 4 instead.

<<<<<<< SEARCH
        # Attention mixes value coordinates identically within each head.
        # Rotate each selected pair, counter-rotate the corresponding output
        # columns, and omit the coefficient made exactly zero.
=======
        # Rotate value coordinates and counter-rotate their output columns.
        # Also quotient projection columns zero through two and column four,
        # leaving optimization-sensitive column three unrestricted.
>>>>>>> REPLACE

<<<<<<< SEARCH
            compact_proj_columns = (
                self.proj_bias_basis.transpose(0, 1) @ proj_weight[:, :2]
            )
            remaining_proj_weight = proj_weight[:, 2:].clone()
=======
            compact_proj_columns = (
                self.proj_bias_basis.transpose(0, 1)
                @ proj_weight[:, (0, 1, 2, 4)]
            )
            remaining_proj_weight = torch.cat(
                (proj_weight[:, 3:4], proj_weight[:, 5:]), dim=1
            ).clone()
>>>>>>> REPLACE

<<<<<<< SEARCH
        compact_columns = self.proj_bias_basis @ self.proj_compact_columns
        proj_weight = torch.cat((compact_columns, self.proj.weight), dim=1)
=======
        compact_columns = self.proj_bias_basis @ self.proj_compact_columns
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

<<<<<<< SEARCH
        # Uniform output components are erased by the final LayerNorm.
        # Quotient columns zero through two plus columns four through seven,
        # retaining column three as an unrestricted optimization coordinate.
        with torch.no_grad():
            weight = self.fc2.weight.detach()
            compact_columns = (
                self.bias_basis.transpose(0, 1)
                @ weight[:, (0, 1, 2, 4, 5, 6, 7)]
            )
            remaining_weight = torch.cat(
                (weight[:, 3:4], weight[:, 8:]), dim=1
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