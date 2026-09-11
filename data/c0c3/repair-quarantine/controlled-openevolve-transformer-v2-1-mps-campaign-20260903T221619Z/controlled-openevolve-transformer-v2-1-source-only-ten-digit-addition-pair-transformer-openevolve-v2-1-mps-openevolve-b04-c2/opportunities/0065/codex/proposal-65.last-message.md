MECHANISM: Four-column MLP residual-output common-mode quotient

HYPOTHESIS: Extending the verified 1,600-parameter design by quotienting the fourth `fc2` output column will produce a 1,599-parameter model with at least 99% accuracy, because its uniform residual-channel component is independently erased by the final LayerNorm.

INTENDED_EDIT: Parameterize the first four `fc2` output columns in the existing seven-dimensional zero-sum basis and leave the remaining eight columns unrestricted; the matrix-aware quotient optimizer already supports the new shape.

EVIDENCE: Successively quotienting one, two, and three `fc2` columns achieved 99.60% at 1,602 parameters, 99.80% at 1,601 parameters, and 99.90% at 1,600 parameters, so extending the same exact function-null quotient to the adjacent fourth column is the closest informative reduction.

<<<<<<< SEARCH
        # A uniform component of any output-weight column produces only a
        # per-token common-mode residual shift, which the final LayerNorm
        # erases. Store the first three columns in the zero-sum output basis.
        with torch.no_grad():
            weight = self.fc2.weight.detach()
            compact_columns = (
                self.bias_basis.transpose(0, 1) @ weight[:, :3]
            )
            remaining_weight = weight[:, 3:].clone()
=======
        # A uniform component of any output-weight column produces only a
        # per-token common-mode residual shift, which the final LayerNorm
        # erases. Store the first four columns in the zero-sum output basis.
        with torch.no_grad():
            weight = self.fc2.weight.detach()
            compact_columns = (
                self.bias_basis.transpose(0, 1) @ weight[:, :4]
            )
            remaining_weight = weight[:, 4:].clone()
>>>>>>> REPLACE