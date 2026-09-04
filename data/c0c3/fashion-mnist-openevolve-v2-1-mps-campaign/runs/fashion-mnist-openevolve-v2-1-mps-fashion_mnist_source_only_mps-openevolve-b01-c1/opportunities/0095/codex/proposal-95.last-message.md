MECHANISM: Low-rate classifier dropout

HYPOTHESIS: Adding 10% dropout to the flattened classifier representation will exceed 9,247 correct predictions by reducing reliance on crop-specific coordinate features while preserving inference-time architecture and ensemble behavior.

INTENDED_EDIT: Insert dropout after the classifier’s normalized hidden layer.

EVIDENCE: Increasing flattened-head capacity regressed to 9,210 correct, while additional spatial capacity regressed to 9,228, suggesting the existing representation needs regularization rather than more parameters.

<<<<<<< SEARCH
            nn.GELU(),
            nn.LayerNorm(80),
            nn.Linear(80, 10),
=======
            nn.GELU(),
            nn.LayerNorm(80),
            nn.Dropout(p=0.1),
            nn.Linear(80, 10),
>>>>>>> REPLACE