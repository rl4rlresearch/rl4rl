MECHANISM: Canonical-view-biased flip ensemble

HYPOTHESIS: Weighting original-image logits 0.60 and flipped-image logits 0.40 will exceed 9,330 correct predictions while retaining the proven 1.15 calibration scale.

INTENDED_EDIT: Bias evaluation-time ensembling toward the canonical image without changing training, parameters, or runtime.

EVIDENCE: Equal-weight calibration changes preserved exactly 9,330 predictions, so improving the primary objective requires changing decision boundaries; the prior 0.60/0.40 attempt timed out and supplied no negative model evidence.

<<<<<<< SEARCH
            logits = 1.05 * (0.5 * (logits + flipped_logits))
=======
            logits = 1.15 * (0.60 * logits + 0.40 * flipped_logits)
>>>>>>> REPLACE