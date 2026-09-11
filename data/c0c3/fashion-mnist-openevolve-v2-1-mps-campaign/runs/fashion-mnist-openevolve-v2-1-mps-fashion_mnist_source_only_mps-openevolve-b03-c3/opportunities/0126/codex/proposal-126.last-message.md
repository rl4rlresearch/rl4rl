MECHANISM: Local interpolation of AdamW second-moment memory

HYPOTHESIS: AdamW beta2=0.959 will exceed 9,283 correct predictions by targeting the interpolated optimum between the verified 9,280-correct beta2=0.95, 9,283-correct beta2=0.96, and 9,278-correct beta2=0.97 results.

INTENDED_EDIT: Change only AdamW beta2 from 0.95 to 0.959 while preserving the verified architecture, batch size, loss, augmentation, learning rate, schedule, and TTA.

EVIDENCE: Beta2=0.96 is the best verified setting at 9,283 correct, while both adjacent tested settings regress; the asymmetric declines from 0.95 and 0.97 place the local quadratic optimum slightly below 0.96.

<<<<<<< SEARCH
        betas=(0.9, 0.95),
=======
        betas=(0.9, 0.959),
>>>>>>> REPLACE