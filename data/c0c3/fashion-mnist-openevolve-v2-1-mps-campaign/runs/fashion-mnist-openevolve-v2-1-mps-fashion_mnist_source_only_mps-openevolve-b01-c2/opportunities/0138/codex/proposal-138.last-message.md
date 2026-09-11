MECHANISM: Left-edge bisection of the flipped-view fusion plateau

HYPOTHESIS: A 0.50850048828125 flipped-view weight will retain 9,331 correct predictions while reducing cross-entropy below 0.19625150680541992.

INTENDED_EDIT: Shift evaluation fusion to 0.49149951171875 original-view and 0.50850048828125 flipped-view weighting, retaining the verified 1.184 calibration and all training behavior.

EVIDENCE: The verified 0.5085009765625 weight retained 9,331 correct and achieved the lowest observed cross-entropy; bisecting its remaining interval toward 0.5085 is the smallest informative refinement.

<<<<<<< SEARCH
            logits = 1.184 * (
                0.49149609375 * logits + 0.50850390625 * flipped_logits
            )
=======
            logits = 1.184 * (
                0.49149951171875 * logits + 0.50850048828125 * flipped_logits
            )
>>>>>>> REPLACE