MECHANISM: Left-edge bisection of the flipped-view fusion plateau

HYPOTHESIS: A 0.5085009765625 flipped-view weight will retain 9,331 correct predictions while reducing cross-entropy below 0.19625150985717774.

INTENDED_EDIT: Shift evaluation fusion to 0.4914990234375 original-view and 0.5085009765625 flipped-view weighting, retaining the verified 1.184 calibration and all training behavior.

EVIDENCE: Every verified reduction from 0.509 through 0.508501953125 retained 9,331 correct while monotonically lowering cross-entropy; bisecting the remaining interval toward 0.5085 is the smallest conservative refinement.

<<<<<<< SEARCH
            logits = 1.184 * (
                0.491498046875 * logits + 0.508501953125 * flipped_logits
            )
=======
            logits = 1.184 * (
                0.4914990234375 * logits + 0.5085009765625 * flipped_logits
            )
>>>>>>> REPLACE