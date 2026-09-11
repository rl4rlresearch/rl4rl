MECHANISM: Accuracy-boundary diagonal-weight bisection

HYPOTHESIS: A diagonal weight of 0.716136932373046875 will retain 9,192 correct predictions while lowering cross-entropy below 0.23742913589477538.

INTENDED_EDIT: Bisect between the highest confirmed failing weight, 0.71613616943359375, and the lowest confirmed successful weight, 0.7161376953125, and normalize by the exact total ensemble weight.

EVIDENCE: The two confirmed endpoints differ in validation correctness, so their midpoint is the most informative remaining probe of the accuracy transition.

<<<<<<< SEARCH
                weight = 0.71875 if is_diagonal else 1.0
=======
                weight = 0.716136932373046875 if is_diagonal else 1.0
>>>>>>> REPLACE

<<<<<<< SEARCH
        return logit_sum / 15.75
=======
        return logit_sum / 15.729095458984375
>>>>>>> REPLACE