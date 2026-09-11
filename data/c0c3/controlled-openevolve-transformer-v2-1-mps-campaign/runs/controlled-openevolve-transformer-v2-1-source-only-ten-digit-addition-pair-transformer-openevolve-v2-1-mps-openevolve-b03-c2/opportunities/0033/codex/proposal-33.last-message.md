MECHANISM: Complete value-bias absorption into the attention projection bias

HYPOTHESIS: Omitting the final retained value-bias degree will produce a 1587-parameter model with at least 99% accuracy because all eight value-bias coordinates have the same projection-bias-absorbable role and the existing optimizer preserves their full-coordinate updates.

INTENDED_EDIT: Store only the eight query-bias parameters, reconstruct the entire value bias as zero, and let the existing quotient-aware clipping and AdamW logic infer and absorb all eight omitted value coordinates.

EVIDENCE: Retaining only value coordinate 1 achieved 99.81% at 1588 parameters, while six-coordinate omission achieved 99.73%; this directly motivates testing the complete value-bias quotient after the retained-coordinate rotation restored accuracy.

<<<<<<< SEARCH
        # Key bias is softmax-invariant. Reuse the query-bias mean for it,
        # and retain all but five value-bias coordinates; the omitted values
        # are absorbed by the downstream projection bias.
        self.qkv.bias = nn.Parameter(torch.empty(2 * d_model - 5))
=======
        # Key bias is softmax-invariant, and every value-bias coordinate can
        # be absorbed by the downstream projection bias. Store only query
        # bias and reconstruct the other two bias vectors in fixed gauges.
        self.qkv.bias = nn.Parameter(torch.empty(d_model))
>>>>>>> REPLACE

<<<<<<< SEARCH
        value_bias = torch.cat(
            (self.qkv.bias[d_model:], self.qkv.bias.new_zeros(5))
        )
=======
        value_bias = self.qkv.bias.new_zeros(d_model)
>>>>>>> REPLACE