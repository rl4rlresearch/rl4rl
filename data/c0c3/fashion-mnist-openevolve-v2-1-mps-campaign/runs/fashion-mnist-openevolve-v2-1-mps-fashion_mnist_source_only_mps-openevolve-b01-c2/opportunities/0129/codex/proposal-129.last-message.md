MECHANISM: Left-edge bisection of flipped-view fusion

HYPOTHESIS: A 0.5085625 flipped-view weight will retain 9,331 correct predictions while reducing cross-entropy below 0.196251579.

INTENDED_EDIT: Shift evaluation fusion from 0.491375/0.508625 to 0.4914375/0.5085625, retaining the verified 1.184 calibration.

EVIDENCE: The verified 0.508625 weight retained 9,331 correct and improved cross-entropy over 0.50875 and 0.509; the prior 0.5085625 attempt timed out and supplied no contrary performance evidence.

<<<<<<< SEARCH
            logits = 1.184 * (0.491375 * logits + 0.508625 * flipped_logits)
=======
            logits = 1.184 * (0.4914375 * logits + 0.5085625 * flipped_logits)
>>>>>>> REPLACE