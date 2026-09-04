MECHANISM: Full-width depthwise-separable feature refinement

HYPOTHESIS: Replacing the expensive final spatial convolution with a depthwise-separable block will finish reliably while exceeding 9,254 correct predictions by preserving 48-channel features and reallocating saved parameters to a wider classifier head.

INTENDED_EDIT: Factor the 48→48 spatial convolution into depthwise 3×3 and pointwise 1×1 stages, then widen the classifier from 88 to 99 hidden units; the model remains below the 250,000-parameter ceiling.

EVIDENCE: The baseline reached 9,254 correct but required 75.95 seconds, and repeated variants timed out; unlike the inconclusive 24-channel bottleneck, this change substantially reduces spatial computation without narrowing intermediate features.

<<<<<<< SEARCH
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
=======
            nn.Conv2d(
                48, 48, kernel_size=3, padding=1, groups=48, bias=False
            ),
            nn.BatchNorm2d(48),
            nn.GELU(),
            nn.Conv2d(48, 48, kernel_size=1, bias=False),
            nn.BatchNorm2d(48),
            nn.GELU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(48 * 7 * 7, 99),
            nn.GELU(),
            nn.Linear(99, 10),
>>>>>>> REPLACE