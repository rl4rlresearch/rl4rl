MECHANISM: Conservative post-ensemble temperature calibration

HYPOTHESIS: Softening evaluation logits with temperature 1.05 will preserve all 9,280 argmax predictions while reducing validation cross-entropy below 0.198076.

INTENDED_EDIT: Divide only the final test-time ensemble logits by 1.05; training and predicted classes remain unchanged.

EVIDENCE: Sparse tail EMA reduced cross-entropy to 0.19631 while changing only one prediction, showing probability quality can improve near the established decision boundary; temperature scaling targets that tie-breaker without altering argmax decisions.

<<<<<<< SEARCH
        return ensemble - math.log(12.0)
=======
        return (ensemble - math.log(12.0)) / 1.05
>>>>>>> REPLACE