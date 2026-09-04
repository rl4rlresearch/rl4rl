MECHANISM: Dense bottleneck capacity expansion

HYPOTHESIS: Widening the proven spatial classifier head from 48 to 56 units will exceed 9,146 correct predictions while remaining below the 250,000-parameter ceiling.

INTENDED_EDIT: Increase only the classifier bottleneck width, preserving batch size 48, convolutional features, augmentation, optimizer, schedule, and inference averaging.

EVIDENCE: The 216,346-parameter spatial-head model reached 9,146 correct, whereas replacing that head with a 239,562-parameter global-average residual model fell to 8,985; this motivates adding capacity to the successful head rather than reallocating it.

<<<<<<< SEARCH
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 48),
            nn.BatchNorm1d(48),
            nn.ReLU(inplace=True),
            nn.Dropout(0.20),
            nn.Linear(48, 10),
        )
=======
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 56),
            nn.BatchNorm1d(56),
            nn.ReLU(inplace=True),
            nn.Dropout(0.20),
            nn.Linear(56, 10),
        )
>>>>>>> REPLACE