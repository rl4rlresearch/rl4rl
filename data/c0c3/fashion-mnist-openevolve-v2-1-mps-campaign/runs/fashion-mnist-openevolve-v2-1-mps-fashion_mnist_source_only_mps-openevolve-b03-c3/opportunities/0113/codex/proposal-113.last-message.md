MECHANISM: Conservative peak learning-rate refinement

HYPOTHESIS: AdamW beta2=0.96 with a 2.5% lower peak learning rate will exceed 9,283 correct predictions by reducing boundary-crossing update noise while retaining sufficient convergence over 522 steps.

INTENDED_EDIT: Preserve the best verified architecture, augmentation, TTA, beta2, and cosine shape while lowering the optimizer and scheduled base learning rate from 2.0e-3 to 1.95e-3.

EVIDENCE: Beta2=0.96 at 2.0e-3 achieved the best 9,283 correct, whereas raising the learning rate to 2.1e-3 reduced cross-entropy but lost six correct predictions; a conservative lower-side test directly brackets the accuracy-optimal learning rate.

<<<<<<< SEARCH
        lr=2.0e-3,
        betas=(0.9, 0.96),
=======
        lr=1.95e-3,
        betas=(0.9, 0.96),
>>>>>>> REPLACE

<<<<<<< SEARCH
    for group in optimizer.param_groups:
        group["lr"] = 2.0e-3 * multiplier
=======
    for group in optimizer.param_groups:
        group["lr"] = 1.95e-3 * multiplier
>>>>>>> REPLACE