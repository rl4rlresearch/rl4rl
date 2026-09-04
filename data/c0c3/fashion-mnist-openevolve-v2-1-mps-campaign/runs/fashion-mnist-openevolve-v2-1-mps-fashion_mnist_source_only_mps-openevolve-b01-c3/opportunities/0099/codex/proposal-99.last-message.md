MECHANISM: Verified center-biased ten-view pooling with temperature calibration

HYPOTHESIS: Using the verified 1.546875× center weight and 1.22775 logit scale will retain 9,287 correct predictions while reducing cross-entropy below the current 0.2067077972.

INTENDED_EDIT: Increase the centered-view pooling weight from 1.5 to 1.546875 with matching normalization, and increase the final evaluation-logit scale from 1.225 to 1.22775.

EVIDENCE: Reference Designs 2 and 3 used this exact evaluation rule and achieved 9,287 correct at 0.2067060093 cross-entropy, strictly improving the current design with the same model and training procedure.

<<<<<<< SEARCH
        pooled_logits = (
            1.5 * offset_logits[0] + offset_logits[1:].sum(dim=0)
        ) / 5.5
        return 1.225 * pooled_logits
=======
        pooled_logits = (
            1.546875 * offset_logits[0] + offset_logits[1:].sum(dim=0)
        ) / 5.546875
        return 1.22775 * pooled_logits
>>>>>>> REPLACE