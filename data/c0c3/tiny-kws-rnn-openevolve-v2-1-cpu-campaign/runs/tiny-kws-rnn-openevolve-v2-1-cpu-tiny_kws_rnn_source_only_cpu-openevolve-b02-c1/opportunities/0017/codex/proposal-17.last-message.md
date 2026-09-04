MECHANISM: Third boundary-frame trimming

HYPOTHESIS: The verified 60+59 GRU will retain at least 85% validation accuracy when processing frames 3–31, while reducing recurrent execution from 30 to 29 steps and lowering total inference MACs below 694,740,230.

INTENDED_EDIT: Skip the first three input frames instead of the first two, preserving the model architecture and training procedure.

EVIDENCE: Processing frames 2–31 achieved 85.64% accuracy at 694,740,230 MACs; the preceding one-frame trim also remained feasible, making one additional boundary trim the most direct test of the minimum viable temporal budget.

<<<<<<< SEARCH
        return list(range(2, available_frames))
=======
        return list(range(3, available_frames))
>>>>>>> REPLACE