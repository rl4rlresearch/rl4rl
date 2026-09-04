MECHANISM: Convolution-to-head parameter reallocation

HYPOTHESIS: Reducing the second convolutional stage from 64 to 63 channels and expanding the dense bottleneck from 58 to 60 units will exceed 9,290 correct predictions by extending the demonstrated benefit of greater dense class-separation capacity while remaining below the parameter ceiling.

INTENDED_EDIT: Reallocate a small amount of convolutional capacity to two additional dense bottleneck units, producing a 249,759-parameter model without changing training or evaluation.

EVIDENCE: Expanding the dense bottleneck from 48 to 58 units produced the current 9,290-correct design, while subsequent dropout, smoothing, augmentation, EMA, and optimizer changes all failed to improve it; this motivates another budget-constrained head expansion.

<<<<<<< SEARCH
            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.SiLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.SiLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 58),
            nn.BatchNorm1d(58),
            nn.SiLU(),
            nn.Dropout(p=0.1),
            nn.Linear(58, 10),
=======
            nn.Conv2d(32, 63, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(63),
            nn.SiLU(),
            nn.Conv2d(63, 63, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(63),
            nn.SiLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(63 * 7 * 7, 60),
            nn.BatchNorm1d(60),
            nn.SiLU(),
            nn.Dropout(p=0.1),
            nn.Linear(60, 10),
>>>>>>> REPLACE