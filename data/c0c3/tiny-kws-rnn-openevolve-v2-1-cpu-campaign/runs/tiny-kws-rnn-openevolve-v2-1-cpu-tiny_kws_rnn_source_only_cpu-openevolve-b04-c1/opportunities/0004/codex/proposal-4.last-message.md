MECHANISM: Boundary-search uniform causal frame decimation

HYPOTHESIS: Sampling 27 uniformly distributed frames will retain at least 85% validation accuracy while reducing recurrent steps and recurrent MACs by approximately 3.6% versus the verified 28-step model.

INTENDED_EDIT: Reduce the evenly spaced full-recording frame schedule from 28 steps to 27 steps.

EVIDENCE: The 26-step model narrowly missed the target at 84.66%, while 28 steps achieved 85.52%; testing 27 steps directly resolves the remaining compute–accuracy boundary.

<<<<<<< SEARCH
        steps = min(28, available_frames)
=======
        steps = min(27, available_frames)
>>>>>>> REPLACE