MECHANISM: Best-verified calibration restoration

HYPOTHESIS: Restoring the best verified float32 logit scale will preserve exactly 9,257 correct predictions and reduce validation cross-entropy from 0.21441890411376954 to approximately 0.2144188995361328.

INTENDED_EDIT: Replace the inferior one-ULP-lower scale with the best verified scale, leaving training and ensembling unchanged.

EVIDENCE: Scale 1.3560298681259155 achieved the best reported validation score, while its immediately lower float32 neighbor 1.356029748916626 produced worse cross-entropy with identical accuracy; the mixed-precision alternatives could not be verified.

<<<<<<< SEARCH
        return 1.356029748916626 * aggregate_logits
=======
        return 1.3560298681259155 * aggregate_logits
>>>>>>> REPLACE