MECHANISM: Faster-adapting AdamW second-moment estimate

HYPOTHESIS: Reducing AdamW beta2 from 0.999 to 0.99 will exceed 9,290 correct predictions by making per-parameter learning rates respond faster during the limited 1,564-update training run.

INTENDED_EDIT: Set AdamW betas to (0.9, 0.99) while preserving the model, exposure, schedule, augmentation, regularization, and EMA ensemble.

EVIDENCE: The batch-size-50 experiment attempted to improve optimization through more updates but timed out, while recent dropout, smoothing, augmentation, and EMA changes failed to beat 9,290; faster moment adaptation tests the same limited-update bottleneck without additional computation.

<<<<<<< SEARCH
        lr=3e-4,
    )
=======
        lr=3e-4,
        betas=(0.9, 0.99),
    )
>>>>>>> REPLACE