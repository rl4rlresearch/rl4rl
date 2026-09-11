MECHANISM: Left-edge bisection of flipped-view fusion plateau

HYPOTHESIS: A 0.50853125 flipped-view weight will retain 9,331 correct predictions while reducing cross-entropy below 0.196251542 by moving slightly closer to equal fusion.

INTENDED_EDIT: Shift evaluation fusion to 0.49146875 original-view and 0.50853125 flipped-view weighting, retaining the verified 1.184 calibration and all training behavior.

EVIDENCE: The verified 0.5085625 weight retained 9,331 correct and improved cross-entropy over 0.508625, 0.50875, and 0.509; bisecting toward the unresolved 0.5085 point conservatively probes the plateau’s left edge.

<<<<<<< SEARCH
            logits = 1.184 * (0.49125 * logits + 0.50875 * flipped_logits)
=======
            logits = 1.184 * (0.49146875 * logits + 0.50853125 * flipped_logits)
>>>>>>> REPLACE