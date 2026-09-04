MECHANISM: Eighth-column MLP residual-output common-mode quotient

HYPOTHESIS: Extending the qualified 1,596-parameter design by quotienting `fc2` column 8 will produce a 1,595-parameter model with at least 99% accuracy, because its uniform output component is erased by the final LayerNorm while column 3 remains unrestricted.

INTENDED_EDIT: Parameterize `fc2` columns 0, 1, 2, and 4 through 8 in the seven-dimensional zero-sum basis, reconstruct their original ordering, and leave column 3 and columns 9 through 11 unrestricted.

EVIDENCE: Reference Design 3 achieved 99.81% accuracy at 1,596 parameters after successfully extending the compacted run through column 7; quotienting adjacent column 8 is the closest incremental reduction using the same exact LayerNorm-null direction.

<<<<<<< SEARCH
        # Uniform output components are erased by the final LayerNorm.
        # Quotient columns zero through two plus columns four and five,
        # retaining column three as an unrestricted optimization coordinate.
        with torch.no_grad():
            weight = self.fc2.weight.detach()
            compact_columns = (
                self.bias_basis.transpose(0, 1)
                @ weight[:, (0, 1, 2, 4, 5)]
            )
            remaining_weight = torch.cat(
                (weight[:, 3:4], weight[:, 6:]), dim=1
            ).clone()
=======
        # Uniform output components are erased by the final LayerNorm.
        # Quotient columns zero through two plus columns four through eight,
        # retaining column three as an unrestricted optimization coordinate.
        with torch.no_grad():
            weight = self.fc2.weight.detach()
            compact_columns = (
                self.bias_basis.transpose(0, 1)
                @ weight[:, (0, 1, 2, 4, 5, 6, 7, 8)]
            )
            remaining_weight = torch.cat(
                (weight[:, 3:4], weight[:, 9:]), dim=1
            ).clone()
>>>>>>> REPLACE