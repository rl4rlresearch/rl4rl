MECHANISM: Complete per-head key-vector gauge anchoring

HYPOTHESIS: A 800-parameter transformer will retain at least 99% accuracy because fixing each head’s remaining key-coordinate anchor exploits Q/K basis-change freedom while preserving the qualified classifier offset, all three final normalization scales, full-rank lexical embedding, and attention routing capacity.

INTENDED_EDIT: Extend the existing Q/K gauge constraint from the first two key channels to all three channels in each head, fixing two additional QKV coefficients at 0.02.

EVIDENCE: The 802-parameter design achieved 99.92%, whereas removing a final normalization scale caused accuracy to collapse to 10.87%; this motivates preserving the classifier path and extending the already-qualified per-head Q/K gauge anchoring instead.

<<<<<<< SEARCH
        selected_key_channels = {
            head * head_dim + offset
            for head in range(n_head)
            for offset in range(min(2, head_dim))
        }
=======
        selected_key_channels = {
            head * head_dim + offset
            for head in range(n_head)
            for offset in range(min(3, head_dim))
        }
>>>>>>> REPLACE