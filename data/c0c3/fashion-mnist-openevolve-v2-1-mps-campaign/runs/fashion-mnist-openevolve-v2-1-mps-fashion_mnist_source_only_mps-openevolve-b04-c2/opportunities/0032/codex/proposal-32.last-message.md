MECHANISM: Longer-horizon batch-normalization statistics

HYPOTHESIS: Setting all four batch-normalization momenta to 0.01 and evaluation calibration to 1.17817 will reproduce the verified 9,277-correct result, exceeding the current 9,273 correct predictions.

INTENDED_EDIT: Apply momentum=0.01 to every batch-normalization layer and use the calibration from the best qualified reference design.

EVIDENCE: Reference Design 2 used the identical architecture and training procedure with these settings and achieved 9,277 correct at 0.208538673 cross-entropy.

<<<<<<< SEARCH
            nn.Conv2d(1, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.GELU(),
            nn.Conv2d(
                128, 128, kernel_size=5, padding=2, groups=64, bias=False
            ),
            nn.BatchNorm2d(128),
=======
            nn.Conv2d(1, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32, momentum=0.01),
            nn.GELU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64, momentum=0.01),
            nn.GELU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128, momentum=0.01),
            nn.GELU(),
            nn.Conv2d(
                128, 128, kernel_size=5, padding=2, groups=64, bias=False
            ),
            nn.BatchNorm2d(128, momentum=0.01),
>>>>>>> REPLACE

<<<<<<< SEARCH
            logits = 1.17810 * logits
=======
            logits = 1.17817 * logits
>>>>>>> REPLACE