MECHANISM: Margin-guided single-gate width compression

HYPOTHESIS: A 92-unit single-gate recurrent state will retain at least 85% validation accuracy on the verified 26-frame schedule while reducing total dense inference MACs by approximately 7.5%, from 473.82M to 438.48M.

INTENDED_EDIT: Reduce the recurrent, temporal-summary, and classifier width from 96 to 92 units while preserving the successful recurrence, readout, schedule, and training procedure.

EVIDENCE: The 96-unit single-gate model achieved 87.73% accuracy, exceeding the requirement by 2.73 percentage points; this substantial margin motivates a moderate four-unit structural compression before altering the proven temporal schedule or gating mechanism.

<<<<<<< SEARCH
        self.hidden_size = 96
=======
        self.hidden_size = 92
>>>>>>> REPLACE