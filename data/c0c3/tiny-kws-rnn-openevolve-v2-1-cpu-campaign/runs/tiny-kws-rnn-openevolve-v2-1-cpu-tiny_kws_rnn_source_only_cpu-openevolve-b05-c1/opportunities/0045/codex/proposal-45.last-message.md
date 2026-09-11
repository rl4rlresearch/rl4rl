MECHANISM: Coverage-preserving 23-step temporal resampling

HYPOTHESIS: Redistributing the 80-unit signed-extrema model across 23 recurrent steps will retain at least 85% validation accuracy while reducing total inference MACs by approximately 4.1%, from 315.05M to 302.01M.

INTENDED_EDIT: Reduce the uniform base schedule from 27 to 26 candidates and retain the established deletion of the first three nonzero candidates, yielding 23 steps with frame 4 and the final frame preserved.

EVIDENCE: The signed-extrema 80-unit model achieved 86.99% accuracy at 24 steps, nearly two percentage points above threshold; prior coverage-preserving resampling successfully reduced the 83-unit model to 24 steps, while the failed schedule was specifically associated with enlarging the initial gap beyond frame 4.

<<<<<<< SEARCH
        base_steps = min(27, available_frames)
        indices = [
            i * (available_frames - 1) // (base_steps - 1)
            for i in range(base_steps)
        ]
        if len(indices) > 26:
            del indices[1:4]
=======
        base_steps = min(26, available_frames)
        indices = [
            i * (available_frames - 1) // (base_steps - 1)
            for i in range(base_steps)
        ]
        if len(indices) > 25:
            del indices[1:4]
>>>>>>> REPLACE