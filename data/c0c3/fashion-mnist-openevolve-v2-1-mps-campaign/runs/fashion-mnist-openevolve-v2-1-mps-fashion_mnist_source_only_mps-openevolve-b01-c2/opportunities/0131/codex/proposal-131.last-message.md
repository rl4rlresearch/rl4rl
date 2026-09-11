MECHANISM: Left-edge bisection of flipped-view fusion plateau

HYPOTHESIS: A 0.50853125 flipped-view weight will retain 9,331 correct predictions while lowering cross-entropy below 0.196251542.

INTENDED_EDIT: Shift evaluation fusion to 0.49146875 original-view and 0.50853125 flipped-view weighting, retaining the verified 1.184 calibration.

EVIDENCE: The verified 0.5085625 weight retained 9,331 correct and improved cross-entropy over every higher tested weight; the prior 0.50853125 attempt timed out and supplied no contrary performance evidence.

<<<<<<< SEARCH
            logits = 1.184 * (0.491 * logits + 0.509 * flipped_logits)
=======
            logits = 1.184 * (
                0.49146875 * logits + 0.50853125 * flipped_logits
            )
>>>>>>> REPLACE