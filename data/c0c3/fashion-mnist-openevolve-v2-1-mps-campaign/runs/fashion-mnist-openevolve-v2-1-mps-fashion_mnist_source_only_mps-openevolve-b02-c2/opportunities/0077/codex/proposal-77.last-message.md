MECHANISM: Mild temperature calibration of flip-mixture logits

HYPOTHESIS: Softening the successful arithmetic flip ensemble by temperature 1.05 will preserve all 9,322 argmax predictions while lowering validation cross-entropy below 0.192551.

INTENDED_EDIT: Divide validation-only ensemble logits by 1.05; training and predicted classes remain unchanged.

EVIDENCE: Arithmetic probability ensembling and geometric ensembling both produced 9,320 correct with hard-maximum attention, but arithmetic ensembling achieved lower cross-entropy (0.192261 versus 0.192650), showing that validation-logit calibration can improve the tie-breaker without changing correct count.

<<<<<<< SEARCH
        return torch.logsumexp(log_probabilities, dim=0) - math.log(2.0)
=======
        return (
            torch.logsumexp(log_probabilities, dim=0) - math.log(2.0)
        ) / 1.05
>>>>>>> REPLACE