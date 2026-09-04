MECHANISM: Accuracy-boundary diagonal-weight bisection

HYPOTHESIS: A diagonal weight of 0.7161346435546875 will retain 9,192 correct predictions while reducing cross-entropy below 0.23742913589477538.

INTENDED_EDIT: Bisect between the highest confirmed failing weight, 0.716131591796875, and the lowest confirmed successful weight, 0.7161376953125, with exact ensemble normalization.

EVIDENCE: The two confirmed endpoints bracket the accuracy transition; the previous verification of this midpoint timed out and therefore supplied no evidence about its prediction count or cross-entropy.

<<<<<<< SEARCH
                weight = 0.7161376953125 if is_diagonal else 1.0
=======
                weight = 0.7161346435546875 if is_diagonal else 1.0
>>>>>>> REPLACE

<<<<<<< SEARCH
        return logit_sum / 15.7291015625
=======
        return logit_sum / 15.7290771484375
>>>>>>> REPLACE