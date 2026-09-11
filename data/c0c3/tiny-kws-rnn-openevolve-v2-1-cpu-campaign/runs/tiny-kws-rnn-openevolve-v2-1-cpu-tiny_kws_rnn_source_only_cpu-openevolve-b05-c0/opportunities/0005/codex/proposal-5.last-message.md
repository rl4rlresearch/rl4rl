MECHANISM: Boundary frame-count refinement

HYPOTHESIS: Processing 27 uniformly distributed frames will achieve at least 85% validation accuracy while reducing recurrent steps and recurrent MACs by approximately 3.6% versus the verified 28-frame design.

INTENDED_EDIT: Reduce the causal frame schedule from 28 to 27 unique increasing indices spanning the complete recording.

EVIDENCE: The 28-frame model achieved 86.13% accuracy while the 26-frame model missed at 83.93%; testing the intervening 27-frame schedule directly locates the lowest viable step count.

<<<<<<< SEARCH
        steps = min(28, available_frames)
=======
        steps = min(27, available_frames)
>>>>>>> REPLACE