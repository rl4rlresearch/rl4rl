MECHANISM: Higher-update stochastic optimization at the verified beta2 optimum

HYPOTHESIS: AdamW beta2=0.96 with batch size 128 will exceed 9,283 correct predictions by providing roughly 50% more optimizer updates and noisier gradients over the same 100,000 examples.

INTENDED_EDIT: Restore the best verified beta2 and reduce batch size from 192 to 128 while preserving the architecture, learning-rate schedule, augmentation, loss, and TTA.

EVIDENCE: Beta2=0.96 at batch size 192 achieved the best result of 9,283 correct; nearby beta2, peak-rate, cosine-floor, representation, and TTA changes regressed, while batch size and its resulting update count remain untested.

<<<<<<< SEARCH
BATCH_SIZE = 192
GRAD_CLIP_NORM = 5.0
=======
BATCH_SIZE = 128
GRAD_CLIP_NORM = 5.0
>>>>>>> REPLACE

<<<<<<< SEARCH
        lr=2.0e-3,
        betas=(0.9, 0.97),
=======
        lr=2.0e-3,
        betas=(0.9, 0.96),
>>>>>>> REPLACE