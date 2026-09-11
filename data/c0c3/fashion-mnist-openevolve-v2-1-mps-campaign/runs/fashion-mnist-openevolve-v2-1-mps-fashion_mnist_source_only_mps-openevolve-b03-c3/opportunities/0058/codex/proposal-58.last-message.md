MECHANISM: Mild terminal label smoothing

HYPOTHESIS: A 0.04→0.01 smoothing schedule will exceed 9,240 correct predictions by combining the empirically optimal 0.04 initial smoothing with modest late regularization that avoids fully hard-label updates.

INTENDED_EDIT: Change only the cosine label-smoothing schedule from 0.05→0.01 to 0.04→0.01.

EVIDENCE: The 0.04→0 schedule achieved the best score with 9,240 correct, while both 0.03→0 and 0.05→0 reduced correctness; the current 0.05→0.01 schedule also reached 9,240, making 0.04→0.01 the informative missing combination.

<<<<<<< SEARCH
    smoothing = 0.03 + 0.02 * math.cos(math.pi * progress)
=======
    smoothing = 0.025 + 0.015 * math.cos(math.pi * progress)
>>>>>>> REPLACE