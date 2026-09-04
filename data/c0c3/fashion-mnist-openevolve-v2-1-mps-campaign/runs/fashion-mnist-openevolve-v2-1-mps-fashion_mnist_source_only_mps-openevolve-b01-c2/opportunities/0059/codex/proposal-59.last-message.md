MECHANISM: Sample-normalized larger-batch optimization

HYPOTHESIS: Increasing the batch size to 80 while scaling all learning rates by 80/64 will complete within the verification limit and match or exceed 9,330 correct predictions.

INTENDED_EDIT: Use batch size 80 and proportionally scale the AdamW initialization and cosine schedule peak learning rates to preserve learning-rate exposure per processed example.

EVIDENCE: The verified design required 80.73 seconds, while subsequent even scale-only variants repeatedly timed out; reducing optimizer steps from roughly 1,564 to 1,250 directly targets this constraint while retaining the proven architecture, loss, augmentation, schedule shape, and evaluation calibration.

<<<<<<< SEARCH
BATCH_SIZE = 64
GRAD_CLIP_NORM = 5.0
=======
BATCH_SIZE = 80
GRAD_CLIP_NORM = 5.0
>>>>>>> REPLACE

<<<<<<< SEARCH
        lr=3.0e-4,
        betas=(0.9, 0.99),
=======
        lr=3.75e-4,
        betas=(0.9, 0.99),
>>>>>>> REPLACE

<<<<<<< SEARCH
    peak_lr = 3.0e-3
    warmup_steps = max(1, int(0.05 * total_steps))
=======
    peak_lr = 3.75e-3
    warmup_steps = max(1, int(0.05 * total_steps))
>>>>>>> REPLACE