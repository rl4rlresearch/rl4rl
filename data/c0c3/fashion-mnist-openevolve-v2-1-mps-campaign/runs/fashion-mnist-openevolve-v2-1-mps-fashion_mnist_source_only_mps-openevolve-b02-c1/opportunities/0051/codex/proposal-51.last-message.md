MECHANISM: Conservative classifier-dropout strengthening

HYPOTHESIS: Raising classifier dropout from 0.15 to 0.17 will exceed 9,286 correct predictions by modestly improving regularization without materially slowing or destabilizing fixed-exposure optimization.

INTENDED_EDIT: Increase the existing classifier dropout probability to 0.17 while preserving the architecture, optimizer, augmentation, TTA, and tail averaging.

EVIDENCE: Reducing dropout from 0.15 to 0.05 lowered validation_correct from 9,280 to 9,265, directly showing that weaker classifier regularization hurts; larger increases only timed out, so a conservative nearby increase remains untested.

<<<<<<< SEARCH
            nn.Dropout(0.15),
=======
            nn.Dropout(0.17),
>>>>>>> REPLACE