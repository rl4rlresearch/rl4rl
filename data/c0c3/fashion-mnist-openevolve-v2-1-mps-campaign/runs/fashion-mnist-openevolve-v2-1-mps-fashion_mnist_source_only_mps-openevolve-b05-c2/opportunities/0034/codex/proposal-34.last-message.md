MECHANISM: Accuracy-boundary diagonal-weight bisection retry

HYPOTHESIS: A diagonal weight of 0.716136455535888671875 will retain 9,192 correct predictions while lowering validation cross-entropy below 0.23742913818359376.

INTENDED_EDIT: Retry the unresolved midpoint between the highest confirmed failing and lowest confirmed successful diagonal weights, with exact ensemble normalization.

EVIDENCE: Weight 0.71613636016845703125 produced 9,191 correct, while 0.7161365509033203125 produced 9,192; verification of their midpoint timed out and supplied no performance evidence.

<<<<<<< SEARCH
                weight = 0.7161365509033203125 if is_diagonal else 1.0
=======
                weight = 0.716136455535888671875 if is_diagonal else 1.0
>>>>>>> REPLACE

<<<<<<< SEARCH
        return logit_sum / 15.7290924072265625
=======
        return logit_sum / 15.729091644287109375
>>>>>>> REPLACE