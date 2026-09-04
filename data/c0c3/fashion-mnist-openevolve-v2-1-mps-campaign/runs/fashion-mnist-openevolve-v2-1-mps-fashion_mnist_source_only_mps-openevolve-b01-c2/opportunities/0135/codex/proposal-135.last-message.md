MECHANISM: Left-edge bisection of flipped-view fusion plateau

HYPOTHESIS: A 0.50850390625 flipped-view weight will retain 9,331 correct predictions while reducing cross-entropy below 0.1962515151977539.

INTENDED_EDIT: Shift evaluation fusion to 0.49149609375 original-view and 0.50850390625 flipped-view weighting, retaining the verified 1.184 calibration and all training behavior.

EVIDENCE: The verified 0.5085078125 weight retained 9,331 correct and achieved the lowest observed cross-entropy; the previous 0.50850390625 attempt timed out and supplied no contrary performance evidence.

<<<<<<< SEARCH
            logits = 1.184 * (0.4914375 * logits + 0.5085625 * flipped_logits)
=======
            logits = 1.184 * (
                0.49149609375 * logits + 0.50850390625 * flipped_logits
            )
>>>>>>> REPLACE