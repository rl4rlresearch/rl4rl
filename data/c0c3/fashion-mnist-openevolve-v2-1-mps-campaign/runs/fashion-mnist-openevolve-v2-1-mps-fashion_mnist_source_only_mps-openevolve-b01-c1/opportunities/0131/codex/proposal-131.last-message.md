MECHANISM: Example-normalized small-batch AdamW

HYPOTHESIS: Halving the batch size while scaling both initial and scheduled learning rates proportionally will exceed 9,262 correct predictions by providing twice as many, finer-grained optimizer updates without increasing the learning-rate integral per presented example.

INTENDED_EDIT: Change the batch size from 128 to 64 and scale AdamW’s initial and peak learning rates from 3.0e-4/3.0e-3 to 1.5e-4/1.5e-3.

EVIDENCE: The best 233,434-parameter architecture and 0.020→0.009 smoothing schedule should be preserved because head widening regressed to 9,221 and nearby smoothing refinements regressed to 9,257 and 9,256; optimization granularity remains an untested dimension that does not add model compute or parameters.

<<<<<<< SEARCH
BATCH_SIZE = 128
=======
BATCH_SIZE = 64
>>>>>>> REPLACE

<<<<<<< SEARCH
        lr=3.0e-4,
=======
        lr=1.5e-4,
>>>>>>> REPLACE

<<<<<<< SEARCH
        group["lr"] = 3.0e-3 * multiplier
=======
        group["lr"] = 1.5e-3 * multiplier
>>>>>>> REPLACE