MECHANISM: Post-ensemble temperature calibration

HYPOTHESIS: Softening the current tail-averaged ensemble with temperature 1.05 will preserve all 9,286 argmax predictions while reducing validation cross-entropy below 0.197145.

INTENDED_EDIT: Divide only the final evaluation ensemble logits by 1.05, leaving training, averaging, and class predictions unchanged.

EVIDENCE: Half-strength tail averaging raised validation_correct to 9,286, while prior sparse averaging reduced cross-entropy to 0.19631; temperature scaling can improve the tie-break metric without changing argmax decisions, and the prior 1.05 attempt timed out without testing the hypothesis.

<<<<<<< SEARCH
        return ensemble - math.log(12.0)
=======
        return (ensemble - math.log(12.0)) / 1.05
>>>>>>> REPLACE