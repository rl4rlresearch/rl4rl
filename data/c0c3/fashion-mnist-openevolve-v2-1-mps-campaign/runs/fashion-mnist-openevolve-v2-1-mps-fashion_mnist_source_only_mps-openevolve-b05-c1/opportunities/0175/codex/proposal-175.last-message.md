MECHANISM: Stronger TTA-agreement-conditioned temperature scaling

HYPOTHESIS: Increasing the agreement coefficient from 0.08 to 0.12 will preserve all 9,360 argmax predictions while lowering validation cross-entropy below 0.18578439636230468.

INTENDED_EDIT: Increase only the strictly positive consensus-conditioned confidence-scaling coefficient from 0.08 to 0.12.

EVIDENCE: Raising the coefficient from 0.04 to 0.08 preserved all 9,360 correct predictions and reduced cross-entropy from 0.18581797180175783 to 0.18578439636230468; the prior 0.12 attempt timed out without producing contrary validation evidence.

<<<<<<< SEARCH
        confidence_scale = torch.exp(0.08 * agreement)
=======
        confidence_scale = torch.exp(0.12 * agreement)
>>>>>>> REPLACE