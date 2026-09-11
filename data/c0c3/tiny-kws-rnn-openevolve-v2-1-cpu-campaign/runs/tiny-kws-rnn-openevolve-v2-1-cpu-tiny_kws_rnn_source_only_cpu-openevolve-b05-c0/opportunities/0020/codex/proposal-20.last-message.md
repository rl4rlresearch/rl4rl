MECHANISM: Incremental causal prefix trimming

HYPOTHESIS: The 58-unit GRU processing the final 29 frames will retain at least 85% validation accuracy while reducing total inference MACs and recurrent steps by approximately 3.3% versus the verified 30-frame model.

INTENDED_EDIT: Omit the first three input frames while preserving the contiguous final 29-frame sequence and all training settings.

EVIDENCE: The final-30-frame model achieved 85.52% accuracy after the 31-frame model achieved 85.77%; removing one additional earliest frame is the smallest structural cost reduction and directly tests the remaining temporal boundary.

<<<<<<< SEARCH
        start = 2 if available_frames > 3 else 0
=======
        start = 3 if available_frames > 4 else 0
>>>>>>> REPLACE