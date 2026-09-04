MECHANISM: Faster short-horizon second-moment adaptation

HYPOTHESIS: AdamW beta2=0.97 will exceed 9,277 correct predictions by continuing the improvement observed from beta2=0.99 to 0.98 while remaining stable over the fixed 522-step run.

INTENDED_EDIT: Reduce AdamW beta2 from its current default of 0.999 to 0.97 while preserving the strongest architecture, augmentation, schedule, and vertical-biased TTA.

EVIDENCE: Beta2=0.99 reduced cross-entropy to 0.2106812 without losing accuracy, and beta2=0.98 further improved to 9,277 correct with 0.2100638 cross-entropy, supporting a conservative continuation of the same optimizer trend.

<<<<<<< SEARCH
        lr=2.0e-3,
    )
=======
        lr=2.0e-3,
        betas=(0.9, 0.97),
    )
>>>>>>> REPLACE