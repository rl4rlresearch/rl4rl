MECHANISM: Refined terminal learning-rate annealing

HYPOTHESIS: Lowering the cosine learning-rate floor from 5% to 4% of the 3e-3 peak will exceed 9,265 correct predictions by retaining more late plasticity than the 3% floor while avoiding the instability observed at 7%.

INTENDED_EDIT: Change only the terminal learning-rate floor from 1.5e-4 to 1.2e-4.

EVIDENCE: The 5% floor achieved 9,265 correct versus 9,262 at 3%, while 7% regressed sharply to 9,252; testing 4% is the most promising local refinement on the less aggressive side of the current optimum.

<<<<<<< SEARCH
        multiplier = 0.05 + 0.95 * 0.5 * (
=======
        multiplier = 0.04 + 0.96 * 0.5 * (
>>>>>>> REPLACE