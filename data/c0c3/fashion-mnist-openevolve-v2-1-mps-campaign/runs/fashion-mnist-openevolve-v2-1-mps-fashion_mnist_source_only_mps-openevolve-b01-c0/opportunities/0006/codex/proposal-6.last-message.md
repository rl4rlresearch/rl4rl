MECHANISM: Convolutional capacity reallocation

HYPOTHESIS: Replacing dense-head capacity with a fifth convolution at 7×7 resolution will exceed 9,177 correct predictions by learning stronger spatial features while retaining nearly the same parameter count.

INTENDED_EDIT: Add a post-pooling 64-channel convolution and narrow the dense bottleneck from 48 to 36 units, reducing total learned parameters from 216,346 to 215,550.

EVIDENCE: The four-convolution redesign improved the objective from 8,928 to 9,138 correct, while the longer-timescale EMA extension regressed to 9,160; this favors testing additional representational depth instead of further temporal ensembling.

<<<<<<< SEARCH
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.SiLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 48),
            nn.BatchNorm1d(48),
            nn.SiLU(),
            nn.Dropout(p=0.1),
            nn.Linear(48, 10),
=======
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.SiLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.SiLU(),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 36),
            nn.BatchNorm1d(36),
            nn.SiLU(),
            nn.Dropout(p=0.1),
            nn.Linear(36, 10),
>>>>>>> REPLACE