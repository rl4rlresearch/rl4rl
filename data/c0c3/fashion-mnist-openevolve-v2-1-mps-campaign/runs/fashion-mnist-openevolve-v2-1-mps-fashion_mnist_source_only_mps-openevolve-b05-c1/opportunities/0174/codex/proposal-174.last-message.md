MECHANISM: Stronger TTA-agreement-conditioned temperature scaling

HYPOTHESIS: Increasing the agreement coefficient from 0.08 to 0.12 will preserve all 9,360 predictions because it applies a strictly positive per-image scalar, while lowering validation cross-entropy below 0.18578439636230468.

INTENDED_EDIT: Increase only the consensus-conditioned confidence-scaling coefficient from 0.08 to 0.12.

EVIDENCE: Raising the coefficient from 0.04 to 0.08 preserved 9,360 correct predictions and reduced cross-entropy from 0.18581797180175783 to 0.18578439636230468, supporting another cautious step in the same direction.

<<<<<<< SEARCH
        confidence_scale = torch.exp(0.08 * agreement)
=======
        confidence_scale = torch.exp(0.12 * agreement)
>>>>>>> REPLACE