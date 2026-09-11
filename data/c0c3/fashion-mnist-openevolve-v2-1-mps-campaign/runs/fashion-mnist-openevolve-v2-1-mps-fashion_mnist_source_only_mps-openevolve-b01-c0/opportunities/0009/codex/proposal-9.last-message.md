MECHANISM: Intermediate smaller-batch optimization with exposure-matched learning rate and EMA

HYPOTHESIS: Batch size 48 will exceed 9,188 correct predictions by providing about 2,084 optimizer updates while retaining sufficient throughput to finish verification; exposure-matched learning rates and EMA decay will preserve the stability of the successful batch-64 design.

INTENDED_EDIT: Reduce batch size from 64 to 48, scale learning rates by √(48/64), and increase EMA decay to 0.9925 to maintain approximately the same averaging horizon in examples.

EVIDENCE: Reducing batch size from 128 to 64 improved validation correct from 9,177 to 9,188, but batch size 32 timed out; batch size 48 tests an intermediate increase in update count with substantially less runtime risk.

<<<<<<< SEARCH
BATCH_SIZE = 64
=======
BATCH_SIZE = 48
>>>>>>> REPLACE

<<<<<<< SEARCH
        lr=3e-4,
=======
        lr=2.6e-4,
>>>>>>> REPLACE

<<<<<<< SEARCH
            decay = 0.99
=======
            decay = 0.9925
>>>>>>> REPLACE

<<<<<<< SEARCH
    start_lr = 3e-4
    peak_lr = 1.8e-3
    end_lr = 7e-5
=======
    start_lr = 2.6e-4
    peak_lr = 1.56e-3
    end_lr = 6.1e-5
>>>>>>> REPLACE