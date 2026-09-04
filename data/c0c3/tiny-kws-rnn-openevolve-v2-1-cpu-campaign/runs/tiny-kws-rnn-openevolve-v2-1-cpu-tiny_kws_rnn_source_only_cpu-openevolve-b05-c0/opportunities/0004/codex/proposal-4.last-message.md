MECHANISM: Uniform 13:16 causal frame subsampling

HYPOTHESIS: Processing 26 evenly distributed frames will retain at least 85% validation accuracy while reducing recurrent steps and recurrent MACs by approximately 7.1% versus the verified 28-frame design.

INTENDED_EDIT: Reduce the frame schedule from 28 to 26 unique increasing indices spanning the first through final input frames.

EVIDENCE: The 24-frame design narrowly missed the target at 84.42%, while 28 frames reached 86.13%; 26 frames directly tests the midpoint and is expected to preserve sufficient temporal evidence at lower cost.

<<<<<<< SEARCH
        steps = min(28, available_frames)
=======
        steps = min(26, available_frames)
>>>>>>> REPLACE