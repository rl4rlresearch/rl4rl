MECHANISM: Downward logit-temperature refinement

HYPOTHESIS: Lowering the positive logit scale from 1.35604 to 1.35603 will preserve 9,257 correct predictions while reducing validation cross-entropy below 0.21441891059875487.

INTENDED_EDIT: Keep the best verified center-view ensemble unchanged and decrease only its final logit scale by 0.00001.

EVIDENCE: Scale 1.35604 improved cross-entropy over 1.35605, while increasing to 1.35606 worsened it; the next equal-sized downward probe is the most informative unresolved calibration test and cannot change class ordering.

<<<<<<< SEARCH
        return 1.35604 * aggregate_logits
=======
        return 1.35603 * aggregate_logits
>>>>>>> REPLACE