MECHANISM: Conservative classifier-dropout increase

HYPOTHESIS: Increasing classifier dropout from 0.15 to 0.18 will exceed 9,286 correct predictions by modestly strengthening regularization without the optimization risk of a larger increase.

INTENDED_EDIT: Raise the existing classifier dropout probability to 0.18 while leaving the architecture, optimizer, augmentation, TTA, and tail averaging unchanged.

EVIDENCE: Reducing dropout from 0.15 to 0.05 lowered validation_correct from 9,280 to 9,265, directly showing that weaker classifier regularization hurts; the previously proposed 0.20 setting timed out, motivating a conservative untested increase.

<<<<<<< SEARCH
            nn.Dropout(0.15),
=======
            nn.Dropout(0.18),
>>>>>>> REPLACE