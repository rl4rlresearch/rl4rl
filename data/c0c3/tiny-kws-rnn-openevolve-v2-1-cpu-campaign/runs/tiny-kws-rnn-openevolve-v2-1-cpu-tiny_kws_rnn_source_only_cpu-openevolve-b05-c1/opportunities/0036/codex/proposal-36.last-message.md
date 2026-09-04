MECHANISM: Boundary-aware single-gate width compression

HYPOTHESIS: An 83-unit single-gate recurrent state will retain at least 85% validation accuracy on the verified 26-frame schedule while reducing total dense inference MACs by approximately 2.1%, from 371.87M to 363.93M.

INTENDED_EDIT: Reduce the recurrent, temporal-summary, and classifier width from 84 to 83 units while preserving the recurrence, readout, frame schedule, and training procedure.

EVIDENCE: The 84-unit model passed at 85.64% accuracy, while every tested single-gate width from 86 through 96 also passed; because the latest result has narrower margin, a one-unit reduction is the most controlled test of the compression boundary.

<<<<<<< SEARCH
        self.hidden_size = 84
=======
        self.hidden_size = 83
>>>>>>> REPLACE