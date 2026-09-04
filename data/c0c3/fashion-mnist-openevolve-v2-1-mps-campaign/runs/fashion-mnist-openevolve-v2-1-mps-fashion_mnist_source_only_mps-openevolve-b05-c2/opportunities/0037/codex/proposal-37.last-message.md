MECHANISM: Local quadratic interpolation of diagonal TTA weighting

HYPOTHESIS: A diagonal weight of 0.71613780975341796875 will retain 9,192 correct predictions while reducing validation cross-entropy below 0.23742913589477538.

INTENDED_EDIT: Move the diagonal ensemble weight to the estimated local minimum and normalize by its exact total ensemble weight.

EVIDENCE: The current weight is bracketed by worse cross-entropies at 0.716137409210205078125 and 0.71613826751708984375; quadratic interpolation of these three measurements places the estimated minimum 40% of the lower-point spacing above the current weight.

<<<<<<< SEARCH
                weight = 0.7161376953125 if is_diagonal else 1.0
=======
                weight = 0.71613780975341796875 if is_diagonal else 1.0
>>>>>>> REPLACE

<<<<<<< SEARCH
        return logit_sum / 15.7291015625
=======
        return logit_sum / 15.72910247802734375
>>>>>>> REPLACE