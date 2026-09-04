MECHANISM: Early-silence temporal pruning

HYPOTHESIS: Omitting frame 3 from the qualified 21-step schedule will retain at least 85% validation accuracy while reducing execution to 20 recurrent steps and total inference MACs to approximately 264,555,520.

INTENDED_EDIT: Add frame 3 to the skipped set for 32-frame inputs while preserving frame 29, both endpoints, and the existing qualified omissions.

EVIDENCE: Removing frame 6 produced a qualified 21-step model at 85.40%, while removing frame 10 produced only 84.29%; this suggests testing another earlier frame before the accuracy-sensitive speech region.

<<<<<<< SEARCH
            if available_frames >= 32:
                skipped.add(available_frames // 32)
            schedule = [index for index in schedule if index not in skipped]
=======
            if available_frames >= 32:
                skipped.add(available_frames // 32)
                skipped.add(3 * available_frames // 32)
            schedule = [index for index in schedule if index not in skipped]
>>>>>>> REPLACE