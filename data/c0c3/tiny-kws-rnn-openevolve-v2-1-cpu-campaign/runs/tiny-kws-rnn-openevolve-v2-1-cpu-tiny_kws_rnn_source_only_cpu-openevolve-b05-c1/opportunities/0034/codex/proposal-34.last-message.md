MECHANISM: Margin-guided single-gate width compression

HYPOTHESIS: An 86-unit single-gate recurrent state will retain at least 85% validation accuracy on the verified 26-frame schedule while reducing total dense inference MACs by approximately 4.1%, from 404.50M to 388.02M.

INTENDED_EDIT: Reduce the recurrent, temporal-summary, and classifier width from 88 to 86 units while preserving the recurrence, readout, frame schedule, and training procedure.

EVIDENCE: The 88-unit model achieved 86.13% accuracy, while the 90- and 92-unit models also exceeded 86%; the stable results across successive two-unit reductions motivate continuing the same controlled compression.

<<<<<<< SEARCH
        self.hidden_size = 88
=======
        self.hidden_size = 86
>>>>>>> REPLACE