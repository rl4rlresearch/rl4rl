MECHANISM: Residual per-view supervision with verified logit calibration

HYPOTHESIS: Capping the ensemble-loss curriculum at 0.90 will exceed 9,330 correct predictions by preserving independent supervision for both views late in training; if correctness ties, the verified 1.15 evaluation scale will reduce cross-entropy.

INTENDED_EDIT: Preserve the proven architecture, optimizer, fixed label smoothing, and symmetric logit fusion, while retaining 10% per-view loss at the curriculum endpoint and restoring the best verified evaluation scale.

EVIDENCE: Equal-logit fusion achieved 9,330 correct, while probability fusion, orientation-biased fusion, EMA, and annealed label smoothing achieved 9,326, 9,319, 9,327, and 9,312 respectively; this motivates keeping the winning inference and target procedures while modestly strengthening view-specific training. Scale 1.15 preserved all 9,330 predictions and improved cross-entropy to 0.196639.

<<<<<<< SEARCH
            logits = 1.05 * (0.5 * (logits + flipped_logits))
=======
            logits = 1.15 * (0.5 * (logits + flipped_logits))
>>>>>>> REPLACE

<<<<<<< SEARCH
    ensemble_weight = 0.5 + 0.5 * step / max(total_steps - 1, 1)
=======
    ensemble_weight = 0.5 + 0.4 * step / max(total_steps - 1, 1)
>>>>>>> REPLACE