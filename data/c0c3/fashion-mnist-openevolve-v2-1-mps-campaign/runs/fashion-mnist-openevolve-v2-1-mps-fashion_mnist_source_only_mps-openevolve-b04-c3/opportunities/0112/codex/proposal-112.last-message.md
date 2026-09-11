MECHANISM: Verified power-mean flip ensembling with calibrated inference temperature

HYPOTHESIS: Using power 0.5831695556640625 and scaling inference logits by 1.03592 will preserve 9,243 correct predictions while reducing cross-entropy to approximately 0.208833005.

INTENDED_EDIT: Restore the highest-scoring verified power-mean order and positive inference-only logit scale; training remains unchanged.

EVIDENCE: Reference Designs 2 and 3 both achieved the best verified score, 9243.413622061835, with this exact inference configuration; an earlier restoration of scale 1.03592 also improved the objective and became an available design.

<<<<<<< SEARCH
            power = 0.583172607421875
            logits = (
=======
            power = 0.5831695556640625
            logits = 1.03592 * (
>>>>>>> REPLACE