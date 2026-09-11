MECHANISM: Narrow late-trajectory weight averaging

HYPOTHESIS: Averaging only the final 10% of training will exceed 9,286 correct predictions by reducing bias from earlier, higher-learning-rate parameters while retaining late-trajectory smoothing.

INTENDED_EDIT: Start sparse tail averaging at 90% rather than 80%, preserving the sampling interval and 50:50 final blend.

EVIDENCE: Half-strength averaging over the final 20% improved validation_correct from 9,280 to 9,286, while full averaging lost a prediction, suggesting smoothing helps but excessive trajectory bias hurts.

<<<<<<< SEARCH
    in_tail = update >= int(0.8 * total_steps)
=======
    in_tail = update >= int(0.9 * total_steps)
>>>>>>> REPLACE