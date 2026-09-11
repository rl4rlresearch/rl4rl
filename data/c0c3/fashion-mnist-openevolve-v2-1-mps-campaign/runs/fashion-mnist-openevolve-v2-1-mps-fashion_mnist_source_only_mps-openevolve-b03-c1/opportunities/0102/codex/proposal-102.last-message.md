MECHANISM: Stronger dense-head dropout regularization

HYPOTHESIS: Increasing both classifier dropout rates will exceed 9,166 correct predictions by reducing overfitting in the parameter-dominant flattened head without changing the successful representation, optimizer, or runtime profile.

INTENDED_EDIT: Increase the pre-hidden dropout from 0.15 to 0.20 and the pre-logit dropout from 0.10 to 0.15.

EVIDENCE: Reducing dropout lowered correctness to 9,141, while architecture replacements and auxiliary heads failed or timed out; this directly tests the evidence-supported opposite direction with no added parameters or computation.

<<<<<<< SEARCH
            nn.Dropout(0.15),
            nn.Linear(96 * 3 * 3, 160),
            nn.GELU(),
            nn.Dropout(0.10),
=======
            nn.Dropout(0.20),
            nn.Linear(96 * 3 * 3, 160),
            nn.GELU(),
            nn.Dropout(0.15),
>>>>>>> REPLACE