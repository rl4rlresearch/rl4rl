MECHANISM: Sixteenth-batch activation-energy screening

HYPOTHESIS: Restricting the step-22 classifier check to the highest-energy sixteenth will retain at least 13 of the 14 observed early exits, preserve validation accuracy at or above 85%, and reduce total inference MACs below 767,462,355.

INTENDED_EDIT: Halve the zero-MAC step-22 candidate pool from one eighth to one sixteenth while leaving all certified checks from step 23 onward unchanged.

EVIDENCE: Highest-energy eighth screening retained all 14 exits from the unscreened step-22 check and achieved 767,462,355 MACs; the previous sixteenth-screening proposal failed only because its search text was ambiguous, not because the design was disproven.

<<<<<<< SEARCH
        candidate_count = max(1, (averaged.shape[0] + 7) // 8)
=======
        candidate_count = max(1, (averaged.shape[0] + 15) // 16)
>>>>>>> REPLACE