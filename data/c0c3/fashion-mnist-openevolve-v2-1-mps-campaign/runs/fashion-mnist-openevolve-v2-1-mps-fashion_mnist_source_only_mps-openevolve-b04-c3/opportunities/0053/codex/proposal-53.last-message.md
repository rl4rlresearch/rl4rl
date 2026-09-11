MECHANISM: Mildly confidence-weighted probability power-mean ensembling

HYPOTHESIS: A 1.25-order power mean of original and flipped probabilities will exceed 9,242 correct predictions by modestly strengthening the confident-view preservation that made arithmetic probability averaging outperform geometric/logit averaging.

INTENDED_EDIT: Replace equal arithmetic probability mixing with a calibrated 1.25-order probability power mean, leaving training and model parameters unchanged.

EVIDENCE: Arithmetic probability averaging achieved 9,242 correct versus 9,240 for logit-space geometric averaging; a mild extrapolation beyond the arithmetic mean directly tests whether greater weight on confident per-view evidence continues that improvement.

<<<<<<< SEARCH
            logits = torch.logaddexp(
                F.log_softmax(logits, dim=1),
                F.log_softmax(flipped_logits, dim=1),
            ) - math.log(2.0)
=======
            power = 1.25
            logits = (
                torch.logaddexp(
                    power * F.log_softmax(logits, dim=1),
                    power * F.log_softmax(flipped_logits, dim=1),
                )
                - math.log(2.0)
            ) / power
>>>>>>> REPLACE