MECHANISM: Margin-guided single-gate width compression

HYPOTHESIS: An 84-unit single-gate recurrent state will retain at least 85% validation accuracy on the verified 26-frame schedule while reducing total dense inference MACs by approximately 4.2%, from 388.02M to 371.87M.

INTENDED_EDIT: Reduce the recurrent, temporal-summary, and classifier width from 86 to 84 units while preserving the successful recurrence, readout, frame schedule, and training procedure.

EVIDENCE: The 86-unit model achieved 87.24% accuracy, the strongest margin among the compressed single-gate designs, while every preceding two-unit reduction from 92 through 86 units remained above 86%; this supports continuing the established two-unit compression.

<<<<<<< SEARCH
        self.hidden_size = 86
=======
        self.hidden_size = 84
>>>>>>> REPLACE