MECHANISM: Finer-grained optimization with batch-scaled learning rate

HYPOTHESIS: Reducing the batch size to 64 while lowering the peak learning rate to 2.0e-3 will exceed 9,257 correct predictions by providing twice as many, less aggressive optimizer updates without altering the best verified architecture.

INTENDED_EDIT: Halve the training batch size and consistently reduce the AdamW and scheduled peak learning rate.

EVIDENCE: The 237,346-parameter model at batch size 128 is the strongest verified design; added capacity and augmentation both regressed, so this isolates optimization granularity while preserving the successful model.

<<<<<<< SEARCH
BATCH_SIZE = 128
GRAD_CLIP_NORM = 2.0
=======
BATCH_SIZE = 64
GRAD_CLIP_NORM = 2.0
>>>>>>> REPLACE

<<<<<<< SEARCH
        lr=3.0e-3,
        betas=(0.9, 0.99),
=======
        lr=2.0e-3,
        betas=(0.9, 0.99),
>>>>>>> REPLACE

<<<<<<< SEARCH
        group["lr"] = 3.0e-3 * multiplier
=======
        group["lr"] = 2.0e-3 * multiplier
>>>>>>> REPLACE