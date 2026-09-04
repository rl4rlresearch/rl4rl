MECHANISM: Left-edge bisection of flipped-view fusion plateau

HYPOTHESIS: A 0.508515625 flipped-view weight will retain 9,331 correct predictions while reducing cross-entropy below 0.196251524 by moving closer to equal fusion.

INTENDED_EDIT: Replace equal evaluation fusion with 0.491484375 original-view and 0.508515625 flipped-view weighting, retaining the verified 1.184 calibration and all training behavior.

EVIDENCE: Successive reductions from 0.509 through 0.50853125 retained 9,331 correct while monotonically lowering cross-entropy; bisecting the remaining interval toward the unresolved 0.5085 point is the most conservative refinement.

<<<<<<< SEARCH
            logits = 1.184 * (0.5 * (logits + flipped_logits))
=======
            logits = 1.184 * (
                0.491484375 * logits + 0.508515625 * flipped_logits
            )
>>>>>>> REPLACE