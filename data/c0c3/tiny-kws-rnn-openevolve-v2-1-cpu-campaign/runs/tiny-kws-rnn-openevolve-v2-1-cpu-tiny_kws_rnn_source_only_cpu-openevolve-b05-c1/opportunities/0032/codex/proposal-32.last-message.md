MECHANISM: Margin-guided single-gate width compression

HYPOTHESIS: A 90-unit single-gate recurrent state will retain at least 85% validation accuracy on the verified 26-frame schedule while reducing total dense inference MACs by approximately 3.9%, from 438.48M to 421.32M.

INTENDED_EDIT: Reduce the recurrent, temporal-summary, and classifier width from 92 to 90 units while preserving the successful recurrence, readout, frame schedule, and training procedure.

EVIDENCE: The 92-unit single-gate model achieved 86.13% accuracy at 438.48M MACs, while the 96-unit version achieved 87.73%; this supports a measured two-unit compression that tests the lower-width boundary without changing the proven temporal or gating design.

<<<<<<< SEARCH
        self.hidden_size = 92
=======
        self.hidden_size = 90
>>>>>>> REPLACE