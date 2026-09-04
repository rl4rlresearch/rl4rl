MECHANISM: Boundary-aware full-rank width compression

HYPOTHESIS: An 80-unit single-gate model will retain at least 85% validation accuracy on the verified 24-step schedule while reducing total dense inference MACs by approximately 2.2%, from 321.63M to 314.52M.

INTENDED_EDIT: Reduce the recurrent state, temporal summaries, and classifier input width from 81 to 80 units while preserving the successful recurrence, readout, schedule, and training procedure.

EVIDENCE: The 81-unit model achieved 85.77% accuracy at 321.63M MACs, while the 82- and 83-unit models also passed on the same 24-step schedule; this supports one more controlled one-unit compression to locate the accuracy boundary.

<<<<<<< SEARCH
        self.hidden_size = 81
=======
        self.hidden_size = 80
>>>>>>> REPLACE