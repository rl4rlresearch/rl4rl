MECHANISM: Accuracy-boundary diagonal-weight bisection retry

HYPOTHESIS: A diagonal weight of 0.7161365509033203125 will retain 9,192 correct predictions while lowering cross-entropy below 0.23742914199829102.

INTENDED_EDIT: Retry the unresolved midpoint between the highest confirmed failing and lowest confirmed successful diagonal weights, with exact ensemble normalization.

EVIDENCE: Weight 0.71613616943359375 produced 9,191 correct, while 0.716136932373046875 produced 9,192; the midpoint’s only verification timed out, so it remains the most informative accuracy-boundary probe.

<<<<<<< SEARCH
                weight = 0.716136932373046875 if is_diagonal else 1.0
=======
                weight = 0.7161365509033203125 if is_diagonal else 1.0
>>>>>>> REPLACE

<<<<<<< SEARCH
        return logit_sum / 15.729095458984375
=======
        return logit_sum / 15.7290924072265625
>>>>>>> REPLACE