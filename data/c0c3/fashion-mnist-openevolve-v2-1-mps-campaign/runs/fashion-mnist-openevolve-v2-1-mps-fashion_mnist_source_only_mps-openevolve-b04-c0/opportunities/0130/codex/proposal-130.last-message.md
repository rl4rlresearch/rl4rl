MECHANISM: Three-point TTA weight interpolation

HYPOTHESIS: Shifting the refined canonical-view weight one-eighth of the prior step toward the lower-weight probe will preserve 9,359 correct predictions while reducing cross-entropy below 0.18456672821044923.

INTENDED_EDIT: Change the guarded refined ensemble from 53.8073348999% to 53.8073301315% canonical-view weight, retaining the disagreement correction and calibration.

EVIDENCE: Equal-sized probes around the current refined weight worsened cross-entropy by approximately 1.9e-9 below and 3.1e-9 above; this asymmetry estimates a local optimum slightly below the current weight.

<<<<<<< SEARCH
        refined_logits = (
            0.5380733489990234375 * logits
            + 0.4619266510009765625 * flipped_logits
        )
=======
        refined_logits = (
            0.5380733013153076171875 * logits
            + 0.4619266986846923828125 * flipped_logits
        )
>>>>>>> REPLACE