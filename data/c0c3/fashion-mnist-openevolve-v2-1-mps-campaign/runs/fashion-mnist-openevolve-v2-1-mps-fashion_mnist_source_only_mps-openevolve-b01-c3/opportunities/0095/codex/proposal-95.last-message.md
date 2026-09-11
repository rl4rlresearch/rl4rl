MECHANISM: Conservative accuracy-boundary refinement for center-biased view pooling

HYPOTHESIS: A 1.55078125× center-offset weight with the calibrated 1.22775 logit scale will retain 9,287 correct predictions while reducing validation cross-entropy below 0.2067060093.

INTENDED_EDIT: Restore the best calibrated scale and move the center weight one quarter of the way from the verified-safe 1.546875× setting toward the accuracy-losing 1.5625× setting.

EVIDENCE: A 1.546875× center weight retained 9,287 correct, while 1.5625× lost one prediction despite lower cross-entropy; 1.55078125× is a conservative unresolved boundary probe, and 1.22775 is the best verified calibration.

<<<<<<< SEARCH
        pooled_logits = (
            1.5 * offset_logits[0] + offset_logits[1:].sum(dim=0)
        ) / 5.5
        return 1.225 * pooled_logits
=======
        pooled_logits = (
            1.55078125 * offset_logits[0] + offset_logits[1:].sum(dim=0)
        ) / 5.55078125
        return 1.22775 * pooled_logits
>>>>>>> REPLACE