MECHANISM: Elevated terminal cosine learning-rate floor

HYPOTHESIS: Restoring the accuracy-optimal 2.0e-3 peak and raising the cosine floor from 5% to 10% will exceed 9,283 correct predictions by sustaining useful late-run optimization without disturbing the proven peak rate.

INTENDED_EDIT: Restore AdamW’s peak learning rate to 2.0e-3 and increase the terminal cosine floor to 10%.

EVIDENCE: Beta2=0.96 with a 2.0e-3 peak and 5% floor achieved the best 9,283 correct, while annealing to zero sharply regressed to 9,252 and worsened cross-entropy, motivating an upper-side test of the terminal floor.

<<<<<<< SEARCH
        lr=2.1e-3,
        betas=(0.9, 0.96),
=======
        lr=2.0e-3,
        betas=(0.9, 0.96),
>>>>>>> REPLACE

<<<<<<< SEARCH
    multiplier = 0.05 + 0.95 * 0.5 * (1.0 + math.cos(math.pi * progress))
    for group in optimizer.param_groups:
        group["lr"] = 2.1e-3 * multiplier
=======
    multiplier = 0.10 + 0.90 * 0.5 * (1.0 + math.cos(math.pi * progress))
    for group in optimizer.param_groups:
        group["lr"] = 2.0e-3 * multiplier
>>>>>>> REPLACE