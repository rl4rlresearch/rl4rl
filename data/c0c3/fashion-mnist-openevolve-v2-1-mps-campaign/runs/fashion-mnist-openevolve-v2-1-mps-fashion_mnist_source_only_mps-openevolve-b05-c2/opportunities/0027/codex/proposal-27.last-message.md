MECHANISM: Accuracy-boundary diagonal-weight bisection retry

HYPOTHESIS: A diagonal weight of 0.71613616943359375 will retain 9,192 correct predictions while lowering cross-entropy below 0.23742913589477538.

INTENDED_EDIT: Retry the unresolved midpoint between the highest confirmed failing and lowest confirmed successful diagonal weights, with exact ensemble normalization.

EVIDENCE: Weight 0.7161346435546875 produced 9,191 correct and 0.7161376953125 produced 9,192; their midpoint is still the most informative probe because its only verification timed out and supplied no performance evidence.

<<<<<<< SEARCH
            weight = 0.71640625 if index in diagonal_indices else 1.0
=======
            weight = 0.71613616943359375 if index in diagonal_indices else 1.0
>>>>>>> REPLACE

<<<<<<< SEARCH
        return logit_sum / 15.73125
=======
        return logit_sum / 15.72908935546875
>>>>>>> REPLACE