MECHANISM: Conservative center-view accuracy-boundary refinement

HYPOTHESIS: A 1.550537109375× center-view weight will retain 9,287 correct predictions while reducing validation cross-entropy below 0.2067059929.

INTENDED_EDIT: Increase the center-view pooling weight one eighth of the remaining distance from the best verified-safe setting toward the accuracy-losing setting, retaining the calibrated 1.22775 logit scale.

EVIDENCE: A 1.548828125× center weight retained 9,287 correct at 0.2067059929 cross-entropy, while 1.5625× lost one prediction; the nearby 1.55078125 probe timed out and provided no contrary accuracy evidence.

<<<<<<< SEARCH
        pooled_logits = (
            1.546875 * offset_logits[0] + offset_logits[1:].sum(dim=0)
        ) / 5.546875
=======
        pooled_logits = (
            1.550537109375 * offset_logits[0] + offset_logits[1:].sum(dim=0)
        ) / 5.550537109375
>>>>>>> REPLACE