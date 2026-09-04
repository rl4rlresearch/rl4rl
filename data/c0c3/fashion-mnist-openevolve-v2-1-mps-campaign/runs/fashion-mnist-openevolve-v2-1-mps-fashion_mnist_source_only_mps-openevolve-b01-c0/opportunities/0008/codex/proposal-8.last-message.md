MECHANISM: Further smaller-batch optimization with exposure-matched learning rate and EMA

HYPOTHESIS: Reducing batch size from 64 to 32 will exceed 9,188 correct predictions by doubling optimizer updates again, while square-root learning-rate scaling and an EMA decay of 0.995 preserve update stability and approximately the same averaging horizon in examples.

INTENDED_EDIT: Use batch size 32, scale all learning-rate schedule values by approximately √½, and increase EMA decay from 0.99 to 0.995.

EVIDENCE: The previous controlled reduction from batch size 128 to 64 improved validation correct from 9,177 to 9,188; this directly motivates testing the same exposure-matched step-doubling mechanism once more.

<<<<<<< SEARCH
BATCH_SIZE = 64
=======
BATCH_SIZE = 32
>>>>>>> REPLACE

<<<<<<< SEARCH
        lr=3e-4,
=======
        lr=2.1e-4,
>>>>>>> REPLACE

<<<<<<< SEARCH
            decay = 0.99
=======
            decay = 0.995
>>>>>>> REPLACE

<<<<<<< SEARCH
    start_lr = 3e-4
    peak_lr = 1.8e-3
    end_lr = 7e-5
=======
    start_lr = 2.1e-4
    peak_lr = 1.27e-3
    end_lr = 5e-5
>>>>>>> REPLACE