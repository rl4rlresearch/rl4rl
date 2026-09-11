MECHANISM: Accuracy-boundary batch-normalization bisection

HYPOTHESIS: Momentum 0.006233984375 will retain 9,279 correct predictions while lowering validation cross-entropy below 0.208617718.

INTENDED_EDIT: Set all four batch-normalization momenta to the midpoint between 0.00623359375 and 0.006234375, preserving every other setting.

EVIDENCE: Momentum 0.00623359375 achieved 9,279 correct, while 0.006234375 lost one prediction despite lower cross-entropy; bisecting this remaining boundary is the most informative next probe.

<<<<<<< SEARCH
            nn.BatchNorm2d(32, momentum=0.0065),
            nn.GELU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64, momentum=0.0065),
            nn.GELU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128, momentum=0.0065),
            nn.GELU(),
            nn.Conv2d(
                128, 128, kernel_size=5, padding=2, groups=64, bias=False
            ),
            nn.BatchNorm2d(128, momentum=0.0065),
=======
            nn.BatchNorm2d(32, momentum=0.006233984375),
            nn.GELU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64, momentum=0.006233984375),
            nn.GELU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128, momentum=0.006233984375),
            nn.GELU(),
            nn.Conv2d(
                128, 128, kernel_size=5, padding=2, groups=64, bias=False
            ),
            nn.BatchNorm2d(128, momentum=0.006233984375),
>>>>>>> REPLACE