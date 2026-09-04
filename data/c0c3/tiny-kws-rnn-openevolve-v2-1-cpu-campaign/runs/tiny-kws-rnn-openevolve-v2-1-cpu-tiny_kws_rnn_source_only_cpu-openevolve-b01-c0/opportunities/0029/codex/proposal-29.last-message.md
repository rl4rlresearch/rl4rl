MECHANISM: Single-step causal schedule trim

HYPOTHESIS: The passing 117-unit additive-readout GRU will retain at least 85% validation accuracy with 24 scheduled frames while reducing total inference MACs by approximately 4% versus the verified 25-step model.

INTENDED_EDIT: Remove one late intermediate frame from the schedule while preserving the first processed frame pattern, endpoint coverage, architecture, and training procedure.

EVIDENCE: The 117-unit, 25-step model achieved 85.767% accuracy, while reducing width to 116 narrowly failed at 84.908%; testing a minimal temporal reduction at the passing width explores a different cost axis with a larger potential MAC improvement.

<<<<<<< SEARCH
        return schedule[1:-3] + schedule[-1:] if len(schedule) > 4 else schedule
=======
        return schedule[1:-4] + schedule[-1:] if len(schedule) > 5 else schedule
>>>>>>> REPLACE