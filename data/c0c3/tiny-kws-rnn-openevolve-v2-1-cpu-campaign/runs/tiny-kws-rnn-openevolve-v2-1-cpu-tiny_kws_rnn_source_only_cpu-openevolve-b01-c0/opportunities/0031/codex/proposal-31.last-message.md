MECHANISM: Early-frame redundancy pruning

HYPOTHESIS: The 117-unit GRU will retain at least 85% validation accuracy with 23 recurrent steps when the second early frame is removed instead of a late intermediate frame, reducing total inference MACs by approximately 4.2%.

INTENDED_EDIT: Preserve the passing 24-frame schedule’s first frame, late intermediate coverage, and final frame, but remove its second processed frame.

EVIDENCE: The 24-step model passed at 85.153%, while the prior 23-step schedule that removed a late intermediate frame fell to 84.294%; this directly motivates testing whether that failure was caused by losing informative late speech rather than by the step count itself.

<<<<<<< SEARCH
        return schedule[1:-4] + schedule[-1:] if len(schedule) > 5 else schedule
=======
        return (
            schedule[1:2] + schedule[3:-4] + schedule[-1:]
            if len(schedule) > 5
            else schedule
        )
>>>>>>> REPLACE