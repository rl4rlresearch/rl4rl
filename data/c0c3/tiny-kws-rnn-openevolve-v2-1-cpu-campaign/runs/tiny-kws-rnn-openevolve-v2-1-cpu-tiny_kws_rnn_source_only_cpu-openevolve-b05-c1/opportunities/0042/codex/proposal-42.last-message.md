MECHANISM: Boundary-aware full-rank width compression

HYPOTHESIS: An 81-unit single-gate model will retain at least 85% validation accuracy on the verified 24-step schedule while reducing total dense inference MACs by approximately 2.2%, from 328.80M to 321.63M.

INTENDED_EDIT: Reduce the recurrent state, temporal summaries, and classifier input width from 82 to 81 units while preserving the successful full-rank recurrence, readout, frame schedule, and training procedure.

EVIDENCE: The 82-unit model passed at 85.64% accuracy, the 83-unit model passed at 86.13% on the same 24-step schedule, and every tested full-rank width from 82 through 96 passed; this supports one more controlled one-unit compression while retaining the architecture whose full-rank hidden interaction outperformed the failed diagonal-memory alternative.

<<<<<<< SEARCH
        self.hidden_size = 82
=======
        self.hidden_size = 81
>>>>>>> REPLACE