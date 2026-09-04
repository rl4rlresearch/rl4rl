MECHANISM: Incremental trailing-window temporal trimming

HYPOTHESIS: The qualified 88-unit dual-readout GRU will retain at least 85% validation accuracy on the most recent 27 frames while reducing total inference MACs below 651,791,360 and recurrent steps from 28 to 27 per example.

INTENDED_EDIT: Omit one additional leading input frame, changing the causal schedule from the most recent 28 frames to the most recent 27 while preserving the model and training procedure.

EVIDENCE: The same 88-unit model qualified at 86.50% with 29 frames and 85.89% with 28 frames; the current 0.89-point margin supports testing the next single-frame structural reduction.

<<<<<<< SEARCH
        start = max(available_frames - 28, 0)
=======
        start = max(available_frames - 27, 0)
>>>>>>> REPLACE