MECHANISM: Margin-guided single-gate width compression

HYPOTHESIS: An 88-unit single-gate recurrent state will retain at least 85% validation accuracy on the verified 26-frame schedule while reducing total dense inference MACs by approximately 4.0%, from 421.32M to 404.50M.

INTENDED_EDIT: Reduce the recurrent, temporal-summary, and classifier width from 90 to 88 units while preserving the successful recurrence, readout, frame schedule, and training procedure.

EVIDENCE: The 90-unit model achieved 86.26% accuracy, and the 92-unit model achieved 86.13%; both exceed the threshold by more than one percentage point, supporting another measured two-unit compression without changing the proven temporal or gating design.

<<<<<<< SEARCH
        self.hidden_size = 90
=======
        self.hidden_size = 88
>>>>>>> REPLACE