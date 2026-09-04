MECHANISM: Left-edge bisection of flipped-view fusion

HYPOTHESIS: A 0.5085625 flipped-view weight will retain 9,331 correct predictions while reducing cross-entropy below 0.196251579 by moving closer to equal fusion.

INTENDED_EDIT: Use 0.4914375 original-view and 0.5085625 flipped-view evaluation weights, retaining the verified 1.184 calibration.

EVIDENCE: Weight 0.508625 achieved 9,331 correct and improved cross-entropy over 0.50875 and 0.509; 0.5085625 is the conservative midpoint toward the unresolved 0.5085 trial.

<<<<<<< SEARCH
            logits = 1.184 * (0.5 * (logits + flipped_logits))
=======
            logits = 1.184 * (
                0.4914375 * logits + 0.5085625 * flipped_logits
            )
>>>>>>> REPLACE