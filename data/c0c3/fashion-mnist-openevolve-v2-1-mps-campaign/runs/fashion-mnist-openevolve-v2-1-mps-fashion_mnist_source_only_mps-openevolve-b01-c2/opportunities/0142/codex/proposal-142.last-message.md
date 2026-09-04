MECHANISM: Upper-half bracketed fusion-weight refinement

HYPOTHESIS: A 0.50850146484375 flipped-view weight will retain 9,331 correct predictions and reduce validation cross-entropy below 0.19625150680541992.

INTENDED_EDIT: Set evaluation fusion to 0.49149853515625 original-view and 0.50850146484375 flipped-view weighting while retaining the verified 1.184 calibration and all training behavior.

EVIDENCE: The best verified weight, 0.5085009765625, lies between worse same-accuracy neighbors at 0.50850048828125 and 0.508501953125. Three lower-midpoint attempts timed out without performance evidence, so bisecting the untested upper half is the most informative remaining refinement.

<<<<<<< SEARCH
            logits = 1.184 * (
                0.49149951171875 * logits + 0.50850048828125 * flipped_logits
            )
=======
            logits = 1.184 * (
                0.49149853515625 * logits + 0.50850146484375 * flipped_logits
            )
>>>>>>> REPLACE