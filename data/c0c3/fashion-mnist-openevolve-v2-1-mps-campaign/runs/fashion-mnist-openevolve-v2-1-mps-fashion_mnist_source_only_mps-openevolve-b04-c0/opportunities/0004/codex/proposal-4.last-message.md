MECHANISM: Convolutional capacity reallocation with coarse spatial retention

HYPOTHESIS: Replacing the oversized 7×7 dense head with an 88-channel third convolutional stage and compact 3×3 spatial head will exceed 9,283 correct predictions while preserving the validated batch-128 optimization regime.

INTENDED_EDIT: Add two batch-normalized 88-channel convolutions and a third pooling stage, then use a 64-unit classifier head; this reallocates parameters from dense layers to spatial feature extraction while remaining below the 250,000-parameter ceiling.

EVIDENCE: The 216,538-parameter deeper CNN reached 9,283 correct, whereas increasing update density and adding translations did not improve it, motivating an architecture-only refinement that preserves the successful training and augmentation settings.

<<<<<<< SEARCH
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 48),
            nn.BatchNorm1d(48),
            nn.GELU(),
            nn.Dropout(p=0.1),
            nn.Linear(48, 10),
        )
=======
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 88, kernel_size=3, padding=1),
            nn.BatchNorm2d(88),
            nn.GELU(),
            nn.Conv2d(88, 88, kernel_size=3, padding=1),
            nn.BatchNorm2d(88),
            nn.GELU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(88 * 3 * 3, 64),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.Dropout(p=0.1),
            nn.Linear(64, 10),
        )
>>>>>>> REPLACE