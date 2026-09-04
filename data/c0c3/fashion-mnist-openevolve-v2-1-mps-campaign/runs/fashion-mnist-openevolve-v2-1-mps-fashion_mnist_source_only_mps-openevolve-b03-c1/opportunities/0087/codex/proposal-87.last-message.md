MECHANISM: Modestly strengthened dense-head dropout

HYPOTHESIS: Increasing classifier dropout from 15%/10% to 20%/15% will exceed 9,166 correct predictions by improving generalization of the parameter-dominant dense head without adding computation or parameters.

INTENDED_EDIT: Raise both classifier dropout probabilities while preserving the architecture, optimizer, augmentation, EMA, and 249,789-parameter count.

EVIDENCE: Reducing dropout to 10%/5% lowered correctness from 9,166 to 9,141, indicating that weaker dense-head regularization is harmful and motivating a direct test in the opposite direction.

<<<<<<< SEARCH
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.15),
            nn.Linear(96 * 3 * 3, 160),
            nn.GELU(),
            nn.Dropout(0.10),
            nn.Linear(160, 10),
        )
=======
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.20),
            nn.Linear(96 * 3 * 3, 160),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(160, 10),
        )
>>>>>>> REPLACE