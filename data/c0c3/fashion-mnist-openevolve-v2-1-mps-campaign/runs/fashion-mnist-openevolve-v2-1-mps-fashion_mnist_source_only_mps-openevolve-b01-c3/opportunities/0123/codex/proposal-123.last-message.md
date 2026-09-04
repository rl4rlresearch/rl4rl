MECHANISM: Conservative center-view boundary refinement

HYPOTHESIS: A 1.557939852073332× center-view weight will retain all 9,287 correct predictions while reducing validation cross-entropy below 0.2067059269.

INTENDED_EDIT: Advance only the centered-view pooling weight one eighth from the best verified value toward the nearest accuracy-losing value, preserving the 1.22775 logit calibration.

EVIDENCE: Reference Design 3 retained 9,287 correct at 1.5578756246377452× with the best verified cross-entropy, while 1.5583894441224401× lost one prediction; this conservative interpolation probes the remaining improvement interval with limited accuracy risk.

<<<<<<< SEARCH
        pooled_logits = (
            1.5544857978820801 * offset_logits[0] + offset_logits[1:].sum(dim=0)
        ) / 5.5544857978820801
=======
        pooled_logits = (
            1.557939852073332 * offset_logits[0] + offset_logits[1:].sum(dim=0)
        ) / 5.557939852073332
>>>>>>> REPLACE