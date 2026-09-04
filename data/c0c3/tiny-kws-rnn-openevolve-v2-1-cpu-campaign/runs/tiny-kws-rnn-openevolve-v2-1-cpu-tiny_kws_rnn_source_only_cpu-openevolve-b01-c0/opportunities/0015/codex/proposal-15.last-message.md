MECHANISM: Mean-and-final readout with one fewer late observation

HYPOTHESIS: The 24-step schedule will retain at least 85% validation accuracy while reducing recurrent MACs by 4% versus the passing 25-step model.

INTENDED_EDIT: Remove frame 28 from the current schedule while preserving all earlier observations, the final frame, and the accuracy-improving concatenated readout.

EVIDENCE: The current 25-step model achieved 86.380% accuracy, providing a 1.38-point margin; its mean-and-final readout improved the same schedule by 1.84 points over mean-only classification, motivating a controlled one-frame reduction.

<<<<<<< SEARCH
        return schedule[1:-3] + schedule[-1:] if len(schedule) > 4 else schedule
=======
        return schedule[1:-4] + schedule[-1:] if len(schedule) > 5 else schedule
>>>>>>> REPLACE