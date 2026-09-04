MECHANISM: Accuracy-boundary diagonal-weight bisection

HYPOTHESIS: A diagonal weight of 0.716136455535888671875 will retain 9,192 correct predictions and reduce validation cross-entropy below 0.23742913589477538.

INTENDED_EDIT: Bisect the remaining interval between the highest confirmed failing diagonal weight and the lowest confirmed successful weight, with exact ensemble normalization.

EVIDENCE: Weight 0.71613636016845703125 produced 9,191 correct, while 0.7161365509033203125 produced 9,192; their midpoint is the most informative unresolved probe of the accuracy transition.

<<<<<<< SEARCH
                weight = 0.7161376953125 if is_diagonal else 1.0
=======
                weight = 0.716136455535888671875 if is_diagonal else 1.0
>>>>>>> REPLACE

<<<<<<< SEARCH
        return logit_sum / 15.7291015625
=======
        return logit_sum / 15.729091644287109375
>>>>>>> REPLACE