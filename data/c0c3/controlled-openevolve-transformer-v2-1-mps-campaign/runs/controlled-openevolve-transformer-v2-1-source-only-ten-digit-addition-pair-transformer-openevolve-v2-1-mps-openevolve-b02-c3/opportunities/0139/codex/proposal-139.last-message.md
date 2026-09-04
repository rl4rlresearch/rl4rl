MECHANISM: Second-head Q/K coordinate-scale gauge fixing

HYPOTHESIS: A 662-parameter transformer will retain at least 99% accuracy because fixing the remaining second-head key-channel scale removes the same Q/K rescaling redundancy whose first-head counterpart produced 99.88% accuracy at 663 parameters.

INTENDED_EDIT: Fix the first input coefficient of the second head’s third key channel at 0.02, allowing its learned query coordinate to absorb the reciprocal scale.

EVIDENCE: The current 663-parameter design achieved 99.88% after fixing the analogous remaining first-head key-channel coefficient, while further relative-bias compression failed; applying the same gauge fixing to the other head is the closest orthogonal reduction.

<<<<<<< SEARCH
        if head_dim > 2:
            selected_key_channels.add(2)
=======
        if head_dim > 2:
            selected_key_channels.update(
                head * head_dim + 2 for head in range(n_head)
            )
>>>>>>> REPLACE