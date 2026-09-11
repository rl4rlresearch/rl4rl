MECHANISM: Conservative center-view accuracy-boundary refinement

HYPOTHESIS: A 1.5583894441224401× center-view weight will retain all 9,287 correct predictions while reducing validation cross-entropy below 0.2067059277.

INTENDED_EDIT: Increase only the centered-view pooling weight and matching normalization denominator, preserving the verified 1.22775 logit calibration.

EVIDENCE: Reference Design 1 achieved the best verified cross-entropy at 1.5578022218542173× while retaining 9,287 correct; advancing one eighth of the remaining interval toward the accuracy-losing 1.5625× boundary continues the consistently successful refinement sequence with limited accuracy risk.

<<<<<<< SEARCH
        pooled_logits = (
            1.5544857978820801 * offset_logits[0] + offset_logits[1:].sum(dim=0)
        ) / 5.5544857978820801
=======
        pooled_logits = (
            1.5583894441224401 * offset_logits[0] + offset_logits[1:].sum(dim=0)
        ) / 5.55838944412244
>>>>>>> REPLACE