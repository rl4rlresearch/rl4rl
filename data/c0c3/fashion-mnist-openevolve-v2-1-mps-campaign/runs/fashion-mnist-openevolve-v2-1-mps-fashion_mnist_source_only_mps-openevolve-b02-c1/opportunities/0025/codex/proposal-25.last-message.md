MECHANISM: Lower-noise cosine schedule tail

HYPOTHESIS: Reducing the terminal learning rate from 1.5e-4 to 3e-5 will exceed 9,280 correct predictions by stabilizing the final iterate while retaining the proven batch-48 optimization trajectory.

INTENDED_EDIT: Lower the cosine schedule floor from 5% to 1% of the peak learning rate, with all other settings unchanged.

EVIDENCE: Sparse tail EMA reduced validation cross-entropy from 0.19808 to 0.19631 while losing only one correct prediction, indicating useful late-training variance reduction; a lower terminal learning rate targets that variance without averaging overhead or replacing the final weights.

<<<<<<< SEARCH
        multiplier = 0.05 + 0.95 * 0.5 * (1.0 + math.cos(math.pi * decay))
=======
        multiplier = 0.01 + 0.99 * 0.5 * (1.0 + math.cos(math.pi * decay))
>>>>>>> REPLACE