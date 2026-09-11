MECHANISM: Passing-schedule second leading-frame ablation

HYPOTHESIS: Removing the next-earliest frame from the passing 27-step schedule will retain at least 85% validation accuracy while reducing recurrent steps and MACs by approximately 3.7%.

INTENDED_EDIT: Preserve 26 indices from the verified schedule by dropping its first two leading frames instead of recomputing uniformly spaced indices.

EVIDENCE: Dropping only frame 0 from the passing 28-frame schedule achieved 85.153% accuracy, whereas recomputing all indices for 27 frames achieved only 84.294%; this supports another controlled leading-frame ablation.

<<<<<<< SEARCH
        return schedule[1:] if len(schedule) > 2 else schedule
=======
        return schedule[2:] if len(schedule) > 3 else schedule
>>>>>>> REPLACE