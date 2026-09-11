MECHANISM: Moderate classifier dropout regularization

HYPOTHESIS: Increasing classifier dropout from 0.15 to 0.20 will exceed 9,286 correct predictions by reducing overfitting during the fixed two-pass exposure without materially slowing optimization.

INTENDED_EDIT: Raise the existing classifier dropout probability to 0.20 while preserving the proven architecture, augmentation, optimizer schedule, TTA, and tail averaging.

EVIDENCE: Reducing dropout from 0.15 to 0.05 lowered validation_correct from 9,280 to 9,265, showing that weaker classifier regularization hurts; the previous 0.20 attempt timed out, so its accuracy effect remains untested.

<<<<<<< SEARCH
            nn.Dropout(0.15),
=======
            nn.Dropout(0.20),
>>>>>>> REPLACE