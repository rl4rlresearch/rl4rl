MECHANISM: Boundary-aware recurrent width compression

HYPOTHESIS: An 82-unit single-gate model will retain at least 85% validation accuracy on the verified 24-step schedule while reducing total dense inference MACs by approximately 2.2%, from 336.06M to 328.80M.

INTENDED_EDIT: Reduce the recurrent, temporal-summary, and classifier width from 83 to 82 units while preserving the successful recurrence, readout, frame schedule, and training procedure.

EVIDENCE: The 83-unit model achieved 86.13% accuracy at 24 steps, and every tested single-gate width from 84 through 96 passed; the one-unit 84-to-83 compression also improved accuracy, supporting another controlled boundary test.

<<<<<<< SEARCH
        self.hidden_size = 83
=======
        self.hidden_size = 82
>>>>>>> REPLACE