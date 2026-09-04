MECHANISM: Seven-column MLP residual-output common-mode quotient

HYPOTHESIS: Extending the verified 1,597-parameter design by quotienting `fc2` column 7 will produce a 1,596-parameter model with at least 99% accuracy, because its uniform output component is erased by the final LayerNorm while column 3 remains unrestricted.

INTENDED_EDIT: Compact `fc2` columns 0, 1, 2, 4, 5, 6, and 7, preserve their original ordering during reconstruction, and retain full-coordinate AdamW optimization through the existing quotient machinery.

EVIDENCE: Quotienting columns 0, 1, 2, 4, 5, and 6 achieved 99.93% accuracy at 1,597 parameters; extending the successful run after column 4 to adjacent column 7 is the closest incremental reduction.

<<<<<<< SEARCH
        # Uniform output components are erased by the final LayerNorm.
        # Quotient columns zero through two plus columns four through six,
        # retaining column three as an unrestricted optimization coordinate.
        with torch.no_grad():
            weight = self.fc2.weight.detach()
            compact_columns = (
                self.bias_basis.transpose(0, 1)
                @ weight[:, (0, 1, 2, 4, 5, 6)]
            )
            remaining_weight = torch.cat(
                (weight[:, 3:4], weight[:, 7:]), dim=1
            ).clone()
=======
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
>>>>>>> REPLACE