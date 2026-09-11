MECHANISM: Incremental redundant early-frame pruning

HYPOTHESIS: Removing one additional densely spaced early frame from the verified 83-unit model will preserve at least 85% validation accuracy while reducing recurrent execution from 25 to 24 steps and total inference MACs by approximately 4%.

INTENDED_EDIT: Expand the schedule deletion from three to four adjacent early indices while preserving the recurrence, temporal readout, training procedure, and first-to-final recording coverage.

EVIDENCE: The 83-unit model improved from 86.26% at 26 steps to 86.63% at 25 steps after pruning a third early frame, leaving 1.63 percentage points of accuracy margin and directly supporting one more controlled schedule reduction.

<<<<<<< SEARCH
        if len(indices) > 27:
            del indices[1:4]
=======
        if len(indices) > 27:
            del indices[1:5]
>>>>>>> REPLACE