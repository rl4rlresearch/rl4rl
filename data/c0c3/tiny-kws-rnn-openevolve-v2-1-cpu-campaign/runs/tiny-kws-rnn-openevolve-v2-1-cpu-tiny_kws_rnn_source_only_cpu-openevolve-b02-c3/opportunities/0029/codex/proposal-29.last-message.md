MECHANISM: Incremental span-preserving temporal thinning

HYPOTHESIS: The qualified 88-unit dual-readout GRU will retain at least 85% validation accuracy using 25 evenly distributed frames across the latest 28-frame window, reducing predicted total inference MACs from 605,316,800 to approximately 582,079,520 and recurrent steps from 26 to 25 per example.

INTENDED_EDIT: Reduce the frame-schedule target from 26 to 25 while preserving the qualified GRU, training procedure, and 28-frame temporal span.

EVIDENCE: The 26-frame span-preserving schedule qualified at 85.03% and achieved lower cross-entropy than the qualified 27-frame contiguous schedule; this indicates that distributed sampling preserves useful temporal coverage and motivates testing the next single-step cost boundary.

<<<<<<< SEARCH
        target_steps = min(26, available_frames)
=======
        target_steps = min(25, available_frames)
>>>>>>> REPLACE