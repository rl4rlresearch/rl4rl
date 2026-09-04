MECHANISM: Moderate-width flattened feature bottleneck

HYPOTHESIS: Expanding the classifier hidden representation from 48 to 54 units will exceed 9,286 correct predictions by preserving more spatial-feature information while remaining comfortably below the parameter and runtime costs of the timed-out 58-unit design.

INTENDED_EDIT: Widen both classifier linear layers to 54 hidden units, increasing learned parameters from 216,250 to 235,132 while leaving the proven convolutional front end, paired-view objective, optimizer, schedule, and EMA unchanged.

EVIDENCE: The best design leaves 33,750 parameters unused, while deeper convolutional alternatives regressed and the 58-unit head timed out; 54 units uses only 18,882 additional low-cost classifier parameters and directly tests the capacity hypothesis with a smaller runtime increase.

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
            nn.Linear(64 * 7 * 7, 54),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(54, 10),
        )
>>>>>>> REPLACE