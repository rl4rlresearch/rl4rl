MECHANISM: Verified power-mean flip ensembling with calibrated inference temperature

HYPOTHESIS: Using power 0.5831695556640625 and scaling ensemble logits by 1.03592 will preserve 9,243 correct predictions while reducing validation cross-entropy from 0.20904548225402833 to approximately 0.20883300514221192.

INTENDED_EDIT: Replace the current unscaled inference ensemble with the best verified power-mean order and positive logit scale; training remains unchanged.

EVIDENCE: Reference Design 3 achieved the highest verified score, 9243.413622061835, with exactly this configuration and the same architecture and training procedure.

<<<<<<< SEARCH
            power = 0.583172607421875
            logits = (
=======
            power = 0.5831695556640625
            logits = 1.03592 * (
>>>>>>> REPLACE