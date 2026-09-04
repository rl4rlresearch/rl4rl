MECHANISM: Compute-aware capacity reallocation

HYPOTHESIS: Reducing convolution widths to 24/48 while widening the classifier bottleneck to 80 will finish reliably by cutting convolutional work roughly 44%, while retaining at least 9,252 correct predictions through a larger spatial classification head.

INTENDED_EDIT: Reallocate capacity from expensive feature-map convolutions to the classifier, producing a 226,002-parameter model while preserving the verified training procedure and ten-view ensemble.

EVIDENCE: Single-view inference and batch-size 64 both still timed out, indicating evaluation overhead and optimizer-step count alone were not the primary issue; reducing training-time convolutional computation directly targets the remaining bottleneck.

<<<<<<< SEARCH
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.08),
            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.08),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 48),
            nn.BatchNorm1d(48),
            nn.ReLU(inplace=True),
            nn.Dropout(0.20),
            nn.Linear(48, 10),
        )
=======
        self.features = nn.Sequential(
            nn.Conv2d(1, 24, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(24),
            nn.ReLU(inplace=True),
            nn.Conv2d(24, 24, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(24),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.08),
            nn.Conv2d(24, 48, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True),
            nn.Conv2d(48, 48, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.08),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(48 * 7 * 7, 80),
            nn.BatchNorm1d(80),
            nn.ReLU(inplace=True),
            nn.Dropout(0.20),
            nn.Linear(80, 10),
        )
>>>>>>> REPLACE