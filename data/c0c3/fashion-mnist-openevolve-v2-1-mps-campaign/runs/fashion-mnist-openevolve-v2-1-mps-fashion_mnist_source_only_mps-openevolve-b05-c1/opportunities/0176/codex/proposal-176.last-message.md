MECHANISM: Incremental TTA-agreement-conditioned temperature scaling

HYPOTHESIS: Increasing the agreement coefficient from 0.12 to 0.16 will preserve all 9,360 argmax predictions while lowering validation cross-entropy below 0.18575621643066406.

INTENDED_EDIT: Increase only the strictly positive consensus-conditioned confidence-scaling coefficient from 0.12 to 0.16.

EVIDENCE: Successive increases from 0.04 to 0.08 and then 0.12 preserved all 9,360 correct predictions while reducing cross-entropy from 0.18581797180175783 to 0.18578439636230468 and then 0.18575621643066406, supporting another equal-sized step.

<<<<<<< SEARCH
        confidence_scale = torch.exp(0.12 * agreement)
=======
        confidence_scale = torch.exp(0.16 * agreement)
>>>>>>> REPLACE