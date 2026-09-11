MECHANISM: Three-row key-projection gauge quotient

HYPOTHESIS: Extending the verified two-row key quotient to the adjacent third-from-last key row will produce a 1,610-parameter model with at least 99% accuracy, because each compacted row independently removes only a position-constant key component that cancels inside attention softmax.

INTENDED_EDIT: Store the final three key-projection rows in independent seven-dimensional zero-sum bases while retaining their reconstructed full-coordinate AdamW updates.

EVIDENCE: The current two-row key quotient achieved 100% accuracy at 1,611 parameters, and the preceding one-row quotient also achieved 100% at 1,612 parameters; this supports the smallest incremental extension of the strongest verified reduction mechanism.

<<<<<<< SEARCH
        # For each compacted key row, one input common-mode coordinate only
        # produces a position-independent key offset and cancels in softmax.
=======
        # For each compacted key row, one input common-mode coordinate only
        # produces a position-independent key offset and cancels in softmax.
>>>>>>> REPLACE

<<<<<<< SEARCH
            compact_key = k_weight[-2:] @ self.key_basis
        self.q_weight = nn.Parameter(q_weight.clone())
        self.k_weight = nn.Parameter(k_weight[:-2].clone())
=======
            compact_key = k_weight[-3:] @ self.key_basis
        self.q_weight = nn.Parameter(q_weight.clone())
        self.k_weight = nn.Parameter(k_weight[:-3].clone())
>>>>>>> REPLACE