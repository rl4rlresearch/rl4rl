MECHANISM: Lower-floor cosine terminal annealing

HYPOTHESIS: Reducing the terminal learning-rate floor from 10% to 2% will exceed 9,316 correct predictions by stabilizing late optimization before EMA averaging.

INTENDED_EDIT: Preserve the successful joint view-fusion model while annealing its learning rate more aggressively near the end of the fixed exposure budget.

EVIDENCE: Joint fusion reached 9,316 correct, whereas classifier widening fell to 9,254 and added coactivation fell to 9,298; the longer-EMA test timed out, so reducing terminal update noise is a focused, runtime-neutral test that adds no capacity.

<<<<<<< SEARCH
    multiplier = 0.10 + 0.90 * 0.5 * (
        1.0 + math.cos(math.pi * decay_progress)
    )
=======
    multiplier = 0.02 + 0.98 * 0.5 * (
        1.0 + math.cos(math.pi * decay_progress)
    )
>>>>>>> REPLACE