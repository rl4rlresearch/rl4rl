MECHANISM: Incremental trailing-frame recurrent step pruning

HYPOTHESIS: Omitting frame 29 in addition to frames 0–2 and 30–31 will retain at least 85% validation accuracy while reducing recurrent execution from 27 to 26 steps and recurrent MACs by approximately 3.70%.

INTENDED_EDIT: Schedule frames 3–28 for standard 32-frame inputs while leaving the verified 18-feature, 98-unit GRU and seven-output zero-sum classifier unchanged.

EVIDENCE: Successively removing trailing frames 31 and 30 preserved passing accuracy at 85.52% and 85.28%, respectively, while removing frame 3 failed; this makes the next trailing frame the best-supported target for another step reduction.

<<<<<<< SEARCH
        return list(range(3, available_frames - 2))
=======
        return list(range(3, available_frames - 3))
>>>>>>> REPLACE