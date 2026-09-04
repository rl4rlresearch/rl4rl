MECHANISM: Incremental trailing-frame recurrent step pruning

HYPOTHESIS: Omitting frame 28 in addition to frames 0–2 and 29–31 will retain at least 85% validation accuracy while reducing recurrent execution from 26 to 25 steps and recurrent MACs by approximately 3.85%.

INTENDED_EDIT: Schedule frames 3–27 for standard 32-frame inputs while leaving the verified 18-feature, 98-unit GRU and seven-output zero-sum classifier unchanged.

EVIDENCE: Successive removal of trailing frames 31, 30, and 29 preserved passing accuracy, with the current 26-step model reaching 85.40%; this makes the next trailing frame the most directly supported target for further cost reduction.

<<<<<<< SEARCH
        return list(range(3, available_frames - 3))
=======
        return list(range(3, available_frames - 4))
>>>>>>> REPLACE