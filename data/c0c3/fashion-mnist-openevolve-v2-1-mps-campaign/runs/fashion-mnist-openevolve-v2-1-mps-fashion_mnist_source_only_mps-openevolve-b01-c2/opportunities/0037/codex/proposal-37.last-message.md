MECHANISM: Canonical-view-biased flip ensemble

HYPOTHESIS: Weighting original-image logits 0.60 and flipped-image logits 0.40 will exceed 9,330 correct predictions by reducing harmful influence from the synthetic view on marginal cases while retaining most flip-ensemble benefit.

INTENDED_EDIT: Preserve the proven training procedure and 1.15 calibration scale, changing only evaluation-time flip-ensemble weights.

EVIDENCE: Scales from 1.05 through 1.15 preserved exactly 9,330 predictions, showing calibration changes cannot improve the primary objective; a controlled asymmetric ensemble is the smallest change that can improve decision boundaries without adding runtime.

<<<<<<< SEARCH
            logits = 1.15 * (0.5 * (logits + flipped_logits))
=======
            logits = 1.15 * (0.60 * logits + 0.40 * flipped_logits)
>>>>>>> REPLACE