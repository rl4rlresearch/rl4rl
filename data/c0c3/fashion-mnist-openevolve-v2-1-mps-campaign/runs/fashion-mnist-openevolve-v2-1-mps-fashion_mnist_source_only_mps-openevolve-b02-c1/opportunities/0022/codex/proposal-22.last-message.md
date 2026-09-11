MECHANISM: Moderately stronger classifier-head dropout

HYPOTHESIS: Increasing classifier dropout from 0.15 to 0.20 will exceed 9,280 correct predictions by improving regularization without changing runtime or parameter count.

INTENDED_EDIT: Raise only the classifier-head dropout probability, preserving the proven architecture, optimizer, schedule, augmentation, batch size, and validation ensemble.

EVIDENCE: Reducing dropout from 0.15 to 0.05 lowered validation_correct from 9,280 to 9,265, indicating that head regularization is beneficial; a conservative increase tests the supported direction.

<<<<<<< SEARCH
            nn.Dropout(0.15),
=======
            nn.Dropout(0.20),
>>>>>>> REPLACE