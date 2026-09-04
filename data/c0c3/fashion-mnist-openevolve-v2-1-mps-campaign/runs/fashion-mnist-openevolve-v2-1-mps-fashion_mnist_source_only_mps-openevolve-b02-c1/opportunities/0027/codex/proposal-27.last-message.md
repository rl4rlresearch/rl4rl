MECHANISM: Conservative late-training noise reduction

HYPOTHESIS: Lowering the cosine learning-rate floor from 5% to 3% will exceed 9,280 correct predictions by stabilizing the final iterate without materially reducing earlier optimization.

INTENDED_EDIT: Reduce only the terminal learning rate from 1.5e-4 to 9e-5, preserving all other training and evaluation behavior.

EVIDENCE: Sparse tail EMA lowered cross-entropy from 0.19808 to 0.19631 while losing only one correct prediction, indicating beneficial late-iterate smoothing; the unscored 1% floor was more aggressive, so 3% is a conservative intermediate test.

<<<<<<< SEARCH
        multiplier = 0.05 + 0.95 * 0.5 * (1.0 + math.cos(math.pi * decay))
=======
        multiplier = 0.03 + 0.97 * 0.5 * (1.0 + math.cos(math.pi * decay))
>>>>>>> REPLACE