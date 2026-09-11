MECHANISM: Stronger classifier-head regularization

HYPOTHESIS: Increasing classifier dropout from 0.10 to 0.15 will exceed 9,249 correct predictions by reducing bottleneck feature co-adaptation.

INTENDED_EDIT: Raise only the classifier-head dropout probability, preserving all other training and inference behavior.

EVIDENCE: Reducing dropout to 0.05 lowered correctness from 9,249 to 9,239, indicating that dropout benefits the primary objective; the prior 0.15 trial timed out and provided no contrary accuracy evidence.

<<<<<<< SEARCH
            nn.Dropout(0.1),
=======
            nn.Dropout(0.15),
>>>>>>> REPLACE