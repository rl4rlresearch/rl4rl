MECHANISM: Single-step causal schedule trim

HYPOTHESIS: The passing 117-unit additive-readout GRU will retain at least 85% validation accuracy with 23 scheduled frames while reducing total inference MACs by approximately 4.2% versus the verified 24-step model.

INTENDED_EDIT: Remove one additional late intermediate frame while preserving the recurrent width, earlier frame pattern, final-frame coverage, and training procedure.

EVIDENCE: The 117-unit model passed at both 25 steps (85.767%) and 24 steps (85.153%), whereas reducing width to 116 narrowly failed; this makes another isolated temporal-step reduction the most informative lower-cost test.

<<<<<<< SEARCH
        return schedule[1:-4] + schedule[-1:] if len(schedule) > 5 else schedule
=======
        return schedule[1:-5] + schedule[-1:] if len(schedule) > 6 else schedule
>>>>>>> REPLACE