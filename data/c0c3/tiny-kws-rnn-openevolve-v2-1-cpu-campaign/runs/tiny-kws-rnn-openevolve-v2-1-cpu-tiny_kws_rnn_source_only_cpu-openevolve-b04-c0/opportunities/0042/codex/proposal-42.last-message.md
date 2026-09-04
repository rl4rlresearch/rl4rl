MECHANISM: Variance-assisted temporal pruning

HYPOTHESIS: A 64-unit GRU with deviation readout processing 26 uniformly spaced frames will achieve at least 85% validation accuracy while reducing dense inference MACs below the verified 64-unit, 27-step model.

INTENDED_EDIT: Reduce the uniform full-utterance frame schedule from 27 to 26 recurrent steps while preserving recurrent width, four-statistic readout, and training procedure.

EVIDENCE: The deviation readout raised the 65-unit, 27-step design from 84.79% to 86.50% and enabled 64 units at 27 steps to qualify at 85.40%; this motivates testing whether its temporal information allows one additional step reduction.

<<<<<<< SEARCH
        target_steps = min(27, available_frames)
=======
        target_steps = min(26, available_frames)
>>>>>>> REPLACE