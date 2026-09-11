MECHANISM: Accuracy-boundary batch-normalization bisection

HYPOTHESIS: Momentum 0.0062336181640625 will retain 9,279 correct predictions while lowering validation cross-entropy below 0.208617718.

INTENDED_EDIT: Set all four batch-normalization momenta to the previously untested midpoint, preserving every other setting.

EVIDENCE: Momentum 0.00623359375 achieved 9,279 correct, while 0.006233642578125 lost one prediction with lower cross-entropy; the prior midpoint patch was malformed, so 0.0062336181640625 remains the most informative boundary probe.

<<<<<<< SEARCH
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32, momentum=0.00623359375),
            nn.GELU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64, momentum=0.00623359375),
            nn.GELU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128, momentum=0.00623359375),
            nn.GELU(),
            nn.Conv2d(
                128, 128, kernel_size=5, padding=2, groups=64, bias=False
            ),
            nn.BatchNorm2d(128, momentum=0.00623359375),
            nn.GELU(),
            nn.MaxPool2d(2),
        )
=======
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32, momentum=0.0062336181640625),
            nn.GELU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64, momentum=0.0062336181640625),
            nn.GELU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128, momentum=0.0062336181640625),
            nn.GELU(),
            nn.Conv2d(
                128, 128, kernel_size=5, padding=2, groups=64, bias=False
            ),
            nn.BatchNorm2d(128, momentum=0.0062336181640625),
            nn.GELU(),
            nn.MaxPool2d(2),
        )
>>>>>>> REPLACE