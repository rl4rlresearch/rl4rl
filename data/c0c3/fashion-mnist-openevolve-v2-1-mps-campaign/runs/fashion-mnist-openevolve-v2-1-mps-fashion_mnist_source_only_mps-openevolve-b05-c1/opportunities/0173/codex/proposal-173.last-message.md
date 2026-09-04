MECHANISM: Stronger TTA-consensus-conditioned temperature scaling

HYPOTHESIS: Increasing the agreement coefficient from 0.04 to 0.08 will retain all 9,360 predictions while further lowering validation cross-entropy below 0.18581797180175783.

INTENDED_EDIT: Double only the strength of the strictly positive per-image confidence scale, preserving training, TTA logits, global calibration, and argmax predictions.

EVIDENCE: Adding the 0.04 agreement-conditioned scale preserved 9,360 correct predictions and improved cross-entropy from the global-scaling result of 0.18585695190429688 to 0.18581797180175783, indicating that greater consensus is positively associated with correctness and motivating a cautious step farther in the same direction.

<<<<<<< SEARCH
        confidence_scale = torch.exp(0.04 * agreement)
=======
        confidence_scale = torch.exp(0.08 * agreement)
>>>>>>> REPLACE