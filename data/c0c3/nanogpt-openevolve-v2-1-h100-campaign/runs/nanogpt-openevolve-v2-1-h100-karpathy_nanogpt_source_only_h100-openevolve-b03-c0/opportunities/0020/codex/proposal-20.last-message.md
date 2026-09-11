MECHANISM: Evidence-guided logit-softcap interpolation

HYPOTHESIS: Setting the logit softcap to 12.5 will achieve `val_bpb < 0.986439` by avoiding the over-compression observed at 7.5 while retaining more regularization than the 15-softcap design.

INTENDED_EDIT: Increase the tanh logit softcap from 10 to 12.5, preserving architecture, optimizer, schedule, batching, and compilation.

EVIDENCE: Softcaps of 7.5, 10, and 15 produced `val_bpb` values of `0.988972`, `0.986439`, and `0.986491`; their quadratic interpolation places the estimated local optimum near 12.5.

<<<<<<< SEARCH
        softcap = 10
=======
        softcap = 12.5
>>>>>>> REPLACE