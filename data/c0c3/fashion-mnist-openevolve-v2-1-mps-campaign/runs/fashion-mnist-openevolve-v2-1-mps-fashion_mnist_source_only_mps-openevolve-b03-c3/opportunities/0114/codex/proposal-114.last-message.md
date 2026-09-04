MECHANISM: Upper-side peak-learning-rate interpolation

HYPOTHESIS: AdamW beta2=0.96 with a 2.025e-3 peak learning rate will retain or exceed 9,283 correct predictions while lowering cross-entropy below 0.2093681.

INTENDED_EDIT: Restore the best verified beta2 and raise both the optimizer and cosine-schedule base learning rate by 1.25%.

EVIDENCE: Beta2=0.96 at 2.0e-3 achieved the best 9,283 correct; 2.1e-3 lowered cross-entropy to 0.2086765 but lost six predictions, while 1.95e-3 worsened both accuracy and cross-entropy, motivating a narrow upper-side interpolation.

<<<<<<< SEARCH
        lr=2.0e-3,
        betas=(0.9, 0.95),
=======
        lr=2.025e-3,
        betas=(0.9, 0.96),
>>>>>>> REPLACE

<<<<<<< SEARCH
        group["lr"] = 2.0e-3 * multiplier
=======
        group["lr"] = 2.025e-3 * multiplier
>>>>>>> REPLACE