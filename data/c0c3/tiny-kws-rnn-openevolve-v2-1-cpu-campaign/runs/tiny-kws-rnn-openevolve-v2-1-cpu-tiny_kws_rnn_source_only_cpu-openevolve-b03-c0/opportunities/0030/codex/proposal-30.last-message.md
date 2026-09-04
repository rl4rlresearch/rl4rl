MECHANISM: Incremental earliest-frame recurrent step pruning

HYPOTHESIS: Omitting frames 0–2 will retain at least 85% validation accuracy while reducing recurrent execution from 30 to 29 steps and recurrent MACs by approximately 3.33%.

INTENDED_EDIT: Schedule frames 3–31 for standard 32-frame inputs while leaving the verified 18-feature, 98-unit GRU and seven-output zero-sum classifier unchanged.

EVIDENCE: Successive removal of frame 0 and then frame 1 preserved passing accuracy at 85.64% and 85.40%, respectively, while each reduced recurrent execution by one step; removing the next-earliest frame is the most direct test of the remaining temporal redundancy.

<<<<<<< SEARCH
        return list(range(2, available_frames))
=======
        return list(range(3, available_frames))
>>>>>>> REPLACE