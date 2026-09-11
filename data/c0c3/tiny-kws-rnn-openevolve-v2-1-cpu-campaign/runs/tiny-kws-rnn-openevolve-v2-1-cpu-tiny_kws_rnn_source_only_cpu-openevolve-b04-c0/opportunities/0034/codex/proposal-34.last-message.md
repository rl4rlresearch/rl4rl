MECHANISM: One-step uniform temporal pruning

HYPOTHESIS: A 65-unit GRU processing 29 uniformly spaced frames will retain at least 85% validation accuracy while reducing exact dense inference MACs by approximately 3.3% versus the verified 30-step model.

INTENDED_EDIT: Reduce the uniform full-utterance frame schedule from 30 to 29 causal recurrent steps while preserving recurrent width, readout, and training procedure.

EVIDENCE: The 65-unit model remained above threshold at 30 steps with 85.77% accuracy after reducing the verified 32-step schedule; this motivates the smallest next probe of temporal redundancy while retaining the recurrent capacity that succeeded where 64-unit variants failed.

<<<<<<< SEARCH
        target_steps = min(30, available_frames)
=======
        target_steps = min(29, available_frames)
>>>>>>> REPLACE