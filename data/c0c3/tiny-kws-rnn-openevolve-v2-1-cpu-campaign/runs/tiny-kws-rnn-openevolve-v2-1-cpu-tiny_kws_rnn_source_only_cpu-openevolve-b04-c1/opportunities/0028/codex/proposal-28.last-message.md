MECHANISM: Boundary-frame temporal pruning

HYPOTHESIS: Processing 29 contiguous near-central frames (indices 2–30) will retain at least 85% validation accuracy while reducing recurrent steps and dense MACs by roughly 3.3% versus the verified 30-step model.

INTENDED_EDIT: Reduce the schedule from 30 to 29 steps and bias the odd-length window toward later frames, removing the earliest remaining boundary frame.

EVIDENCE: The 58-unit GRU improved from 85.15% at 31 steps to 85.77% at 30 central steps, while dropping the original earliest boundary frame already passed; this supports testing one further boundary removal without reducing recurrent capacity.

<<<<<<< SEARCH
        steps = min(30, available_frames)
        start = (available_frames - steps) // 2
=======
        steps = min(29, available_frames)
        start = (available_frames - steps + 1) // 2
>>>>>>> REPLACE