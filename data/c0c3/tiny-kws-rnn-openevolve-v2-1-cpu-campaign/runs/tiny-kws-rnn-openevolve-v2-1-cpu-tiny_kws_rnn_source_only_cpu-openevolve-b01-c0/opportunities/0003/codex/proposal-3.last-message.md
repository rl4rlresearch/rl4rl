MECHANISM: Uniform 27-step causal frame subsampling

HYPOTHESIS: Processing 27 evenly distributed frames will retain at least 85% validation accuracy while reducing recurrent steps and recurrent MACs by approximately 3.6% versus the verified 28-step model.

INTENDED_EDIT: Reduce the frame schedule from 28 to 27 unique, increasing indices spanning the complete recording.

EVIDENCE: The 28-step model achieved 85.276% accuracy, while 24 steps achieved 84.417%; a one-step reduction is the smallest structural cost reduction and the most conservative probe within this observed boundary.

<<<<<<< SEARCH
        steps = min(28, available_frames)
=======
        steps = min(27, available_frames)
>>>>>>> REPLACE