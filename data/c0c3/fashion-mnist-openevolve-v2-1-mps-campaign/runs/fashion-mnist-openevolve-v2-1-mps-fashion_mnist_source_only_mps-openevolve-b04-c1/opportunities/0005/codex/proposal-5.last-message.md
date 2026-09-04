MECHANISM: Small-batch optimization with square-root learning-rate scaling

HYPOTHESIS: Increasing optimizer steps from 782 to about 1,563 while scaling the learning rate to 2.1e-3 will exceed 9,212 correct validation predictions.

INTENDED_EDIT: Halve batch size to 64 and reduce the AdamW base and scheduled learning rates for stable, more frequent updates.

EVIDENCE: Expanding the model to 249,934 parameters regressed to 9,196 correct, while the successful 216,250-parameter design still receives only 782 updates; this motivates spending computation on optimization rather than additional capacity.

<<<<<<< SEARCH
BATCH_SIZE = 128
=======
BATCH_SIZE = 64
>>>>>>> REPLACE

<<<<<<< SEARCH
        lr=3.0e-3,
        betas=(0.9, 0.99),
=======
        lr=2.1e-3,
        betas=(0.9, 0.99),
>>>>>>> REPLACE

<<<<<<< SEARCH
        group["lr"] = 3.0e-3 * multiplier
=======
        group["lr"] = 2.1e-3 * multiplier
>>>>>>> REPLACE