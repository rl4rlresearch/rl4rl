MECHANISM: Wider flattened feature bottleneck

HYPOTHESIS: Widening the classifier hidden layer from 48 to 58 units will exceed 9,286 correct predictions by using the remaining parameter budget to preserve more spatial-feature information without adding expensive convolutional computation.

INTENDED_EDIT: Expand both classifier linear layers to a 58-unit hidden representation, increasing learned parameters from 216,250 to approximately 247,710 while retaining the proven architecture, paired-view objective, optimizer, and EMA.

EVIDENCE: The best design uses only 216,250 of 250,000 permitted parameters, while deeper convolutional and attention alternatives regressed or timed out; widening the compact flattened head tests unused low-cost capacity without repeating those expensive architectural changes.

<<<<<<< SEARCH
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 48),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(48, 10),
        )
=======
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 58),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(58, 10),
        )
>>>>>>> REPLACE