MECHANISM: Single-frame causal temporal subsampling

HYPOTHESIS: The verified 58-unit GRU will retain at least 85% validation accuracy when processing 31 evenly distributed frames, while reducing total dense inference MACs below the failed 57-unit, 32-step model and reducing executed recurrent steps by 3.125%.

INTENDED_EDIT: Change the causal frame schedule from 32 to 31 evenly spaced frames while preserving the verified recurrent width, pooling, and training procedure.

EVIDENCE: The 58-unit, 32-step model passed at 85.40% accuracy, whereas reducing width to 57 failed; removing one of 32 highly adjacent frames tests temporal redundancy while retaining the full verified recurrent capacity.

<<<<<<< SEARCH
        steps = min(32, available_frames)
=======
        steps = min(31, available_frames)
>>>>>>> REPLACE