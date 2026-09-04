MECHANISM: Moderate classifier dropout regularization

HYPOTHESIS: Increasing classifier dropout from 0.15 to 0.20 will exceed 9,286 correct predictions by reducing overfitting without materially impairing optimization during the fixed exposure.

INTENDED_EDIT: Raise the existing classifier dropout probability to 0.20 while preserving the architecture, optimizer, augmentation, tail averaging, and runtime profile.

EVIDENCE: Reducing dropout from 0.15 to 0.05 lowered validation_correct from 9,280 to 9,265, directly indicating that weaker classifier regularization hurts; a modest increase tests the favorable direction with negligible computational overhead.

<<<<<<< SEARCH
            nn.Dropout(0.15),
=======
            nn.Dropout(0.20),
>>>>>>> REPLACE