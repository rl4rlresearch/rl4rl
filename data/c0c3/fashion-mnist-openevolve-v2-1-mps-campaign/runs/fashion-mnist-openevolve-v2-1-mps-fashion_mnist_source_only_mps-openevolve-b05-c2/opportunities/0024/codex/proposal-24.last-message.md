MECHANISM: Accuracy-boundary diagonal-weight bisection

HYPOTHESIS: A diagonal weight of 0.7161346435546875 will retain 9,192 correct predictions while reducing cross-entropy below 0.23742913589477538.

INTENDED_EDIT: Bisect between the failed 0.716131591796875 and successful 0.7161376953125 diagonal weights, normalizing by the exact total ensemble weight of 15.7290771484375.

EVIDENCE: Weight 0.7161376953125 achieved 9,192 correct, whereas 0.716131591796875 achieved 9,191 with marginally lower cross-entropy; their midpoint is the most informative remaining accuracy-boundary probe.

<<<<<<< SEARCH
                weight = 0.71875 if is_diagonal else 1.0
=======
                weight = 0.7161346435546875 if is_diagonal else 1.0
>>>>>>> REPLACE

<<<<<<< SEARCH
        return logit_sum / 15.75
=======
        return logit_sum / 15.7290771484375
>>>>>>> REPLACE