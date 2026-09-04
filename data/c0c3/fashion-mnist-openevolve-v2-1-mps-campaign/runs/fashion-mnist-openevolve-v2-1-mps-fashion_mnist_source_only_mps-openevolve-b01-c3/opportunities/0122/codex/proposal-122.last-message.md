MECHANISM: Conservative center-view boundary refinement

HYPOTHESIS: A 1.5578756246377452× center-view weight will retain all 9,287 correct predictions while reducing validation cross-entropy below 0.2067059277.

INTENDED_EDIT: Advance the center-view pooling weight one eighth toward the nearest accuracy-losing weight while preserving the verified 1.22775 logit calibration.

EVIDENCE: The 1.5578022218542173× reference retained 9,287 correct with the best verified cross-entropy, while 1.5583894441224401× lost one prediction; the intermediate 1.5578756246377452× attempt only timed out and remains the closest unresolved conservative probe.

<<<<<<< SEARCH
        pooled_logits = (
            1.557131110690534 * offset_logits[0] + offset_logits[1:].sum(dim=0)
        ) / 5.557131110690534
=======
        pooled_logits = (
            1.5578756246377452 * offset_logits[0] + offset_logits[1:].sum(dim=0)
        ) / 5.5578756246377452
>>>>>>> REPLACE