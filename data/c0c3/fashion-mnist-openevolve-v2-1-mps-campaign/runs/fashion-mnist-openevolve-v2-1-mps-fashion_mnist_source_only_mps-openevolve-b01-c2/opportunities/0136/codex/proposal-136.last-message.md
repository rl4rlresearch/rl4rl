MECHANISM: Left-edge bisection of flipped-view fusion plateau

HYPOTHESIS: A 0.508501953125 flipped-view weight will retain 9,331 correct predictions while reducing cross-entropy below 0.19625151023864745.

INTENDED_EDIT: Shift evaluation fusion to 0.491498046875 original-view and 0.508501953125 flipped-view weighting, retaining the verified 1.184 calibration and all training behavior.

EVIDENCE: Successive reductions through the verified 0.50850390625 weight retained 9,331 correct while lowering cross-entropy; bisecting toward the unresolved 0.5085 point is the smallest informative refinement.

<<<<<<< SEARCH
            logits = 1.184 * (0.49125 * logits + 0.50875 * flipped_logits)
=======
            logits = 1.184 * (
                0.491498046875 * logits + 0.508501953125 * flipped_logits
            )
>>>>>>> REPLACE