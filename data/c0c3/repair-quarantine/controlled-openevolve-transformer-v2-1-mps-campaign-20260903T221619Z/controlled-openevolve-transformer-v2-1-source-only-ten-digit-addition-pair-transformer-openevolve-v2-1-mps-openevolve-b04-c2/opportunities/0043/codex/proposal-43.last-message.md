MECHANISM: Complete per-head disjoint value-basis gauge fixing

HYPOTHESIS: Gauge-fixing the remaining disjoint value-channel pair in the second attention head will reduce the verified 1,614-parameter model to 1,613 parameters while retaining at least 99% accuracy.

INTENDED_EDIT: Apply the compensated value-channel rotation to both disjoint channel pairs in every four-dimensional attention head, removing one additional learned QKV coefficient.

EVIDENCE: The current design achieved 100% accuracy after adding the second disjoint pair in the first head; applying the same exact intra-head symmetry to the untouched pair in the second head is the smallest direct extension.

<<<<<<< SEARCH
        # Fix one value-basis direction per head, plus a second disjoint
        # direction in the first head when four value channels are available.
        value_basis_pairs = [(head, 0) for head in range(n_head)]
        if self.head_dim >= 4:
            value_basis_pairs.append((0, 2))
=======
        # Fix two disjoint value-basis directions in every head when four
        # value channels are available, otherwise fix the single pair.
        value_basis_pairs = [(head, 0) for head in range(n_head)]
        if self.head_dim >= 4:
            value_basis_pairs.extend((head, 2) for head in range(n_head))
>>>>>>> REPLACE