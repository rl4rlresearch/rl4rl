MECHANISM: Single-frame endpoint-preserving temporal compression

HYPOTHESIS: The 81-unit GRU processing 31 evenly distributed frames will retain at least 85% validation accuracy while reducing recurrent MACs and executed steps by approximately 3.1% versus the verified 32-step model.

INTENDED_EDIT: Reduce the causal frame schedule from 32 to 31 unique increasing indices while preserving both recording endpoints, the mean-plus-terminal readout, and training procedure.

EVIDENCE: The 81-unit 32-step model passed at 85.64% accuracy, while 24-frame processing has previously passed at higher width; with the recurrent-width boundary already isolated between 80 and 81 units, removing one frame is the most cautious probe of the remaining temporal redundancy.

<<<<<<< SEARCH
        steps = min(32, available_frames)
=======
        steps = min(31, available_frames)
>>>>>>> REPLACE