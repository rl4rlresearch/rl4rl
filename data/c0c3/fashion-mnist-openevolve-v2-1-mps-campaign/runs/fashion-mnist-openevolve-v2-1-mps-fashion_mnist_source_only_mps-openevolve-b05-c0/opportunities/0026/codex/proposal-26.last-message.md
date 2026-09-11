MECHANISM: Compute-neutral feature-capacity reallocation

HYPOTHESIS: Reallocating parameters from the oversized dense head into wider convolutional features will exceed 9,254 correct predictions by learning richer spatial representations while reducing per-image multiply-accumulates.

INTENDED_EDIT: Widen feature channels from 24/48/48 to 32/64/64, move the second pooling operation before the final convolution, and narrow the classifier hidden layer from 88 to 60; this uses 244,894 learned parameters.

EVIDENCE: The best design assigns 207,954 of 239,634 parameters to its dense head, while runtime-increasing experiments repeatedly timed out. This redistribution increases convolutional capacity by roughly 77% while reducing approximate convolution-plus-linear compute from 6.47M to 5.83M operations per image.

<<<<<<< SEARCH
        self.features = nn.Sequential(
            nn.Conv2d(1, 24, kernel_size=3, padding=1),
            nn.BatchNorm2d(24),
            nn.GELU(),
            nn.MaxPool2d(2),
            nn.Conv2d(24, 48, kernel_size=3, padding=1),
            nn.BatchNorm2d(48),
            nn.GELU(),
            nn.Conv2d(48, 48, kernel_size=3, padding=1),
            nn.BatchNorm2d(48),
            nn.GELU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(48 * 7 * 7, 88),
            nn.GELU(),
            nn.Linear(88, 10),
        )
=======
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.GELU(),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 60),
            nn.GELU(),
            nn.Linear(60, 10),
        )
>>>>>>> REPLACE