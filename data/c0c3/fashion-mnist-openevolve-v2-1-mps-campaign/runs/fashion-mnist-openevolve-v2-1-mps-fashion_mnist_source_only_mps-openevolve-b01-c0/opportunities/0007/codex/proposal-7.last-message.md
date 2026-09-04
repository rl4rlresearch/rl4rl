MECHANISM: Smaller-batch optimization with exposure-matched EMA

HYPOTHESIS: Doubling optimizer steps from 782 to about 1,563 while applying square-root learning-rate scaling will exceed 9,177 correct predictions by improving optimization under the fixed exposure without destabilizing updates.

INTENDED_EDIT: Reduce batch size from 128 to 64, scale learning rates by approximately √½, and increase EMA decay from 0.98 to 0.99 so its averaging horizon remains approximately constant in examples.

EVIDENCE: The largest prior gain coincided with doubling optimizer updates from 392 to 782, whereas additional EMA timescales and convolutional reallocation regressed; another controlled increase in update opportunities is therefore the most informative remaining lever.

<<<<<<< SEARCH
BATCH_SIZE = 128
GRAD_CLIP_NORM = 2.0
=======
BATCH_SIZE = 64
GRAD_CLIP_NORM = 2.0
>>>>>>> REPLACE

<<<<<<< SEARCH
        lr=4e-4,
    )
=======
        lr=3e-4,
    )
>>>>>>> REPLACE

<<<<<<< SEARCH
            decay = 0.98
            with torch.no_grad():
=======
            decay = 0.99
            with torch.no_grad():
>>>>>>> REPLACE

<<<<<<< SEARCH
    start_lr = 4e-4
    peak_lr = 2.5e-3
    end_lr = 1e-4
=======
    start_lr = 3e-4
    peak_lr = 1.8e-3
    end_lr = 7e-5
>>>>>>> REPLACE