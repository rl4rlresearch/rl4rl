MECHANISM: Near-ceiling final-stage channel expansion

HYPOTHESIS: Expanding the final convolutional stage from 96 to 106 channels will exceed 9,280 correct predictions by increasing learned feature capacity with minimal additional runtime.

INTENDED_EDIT: Widen the final feature stage and matching classifier input, increasing parameters from 232,682 to 249,982 while preserving every proven training and evaluation setting.

EVIDENCE: The batch-48 design completed in 73.86 seconds and achieved 9,280 correct, while slower optimization and residual redesigns timed out; this controlled expansion uses nearly all remaining parameter headroom with only about 1–2% additional convolutional work.

<<<<<<< SEARCH
            nn.Conv2d(64, 96, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(96),
            nn.GELU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(96 * 3 * 3, 128),
=======
            nn.Conv2d(64, 106, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(106),
            nn.GELU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(106 * 3 * 3, 128),
>>>>>>> REPLACE