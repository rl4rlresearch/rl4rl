MECHANISM: Accuracy-boundary batch-normalization bisection

HYPOTHESIS: Momentum 0.00623359375 will retain 9,279 correct predictions while lowering validation cross-entropy below 0.208617760.

INTENDED_EDIT: Set all four batch-normalization momenta to the midpoint between the successful 0.0062328125 setting and the one-error 0.006234375 setting.

EVIDENCE: Momentum 0.0062328125 achieved 9,279 correct with cross-entropy 0.208617760, while 0.006234375 lost one prediction despite lower cross-entropy; their midpoint is the most informative remaining boundary probe.

<<<<<<< SEARCH
            nn.Conv2d(1, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32, momentum=0.0061),
            nn.GELU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64, momentum=0.0061),
            nn.GELU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128, momentum=0.0061),
            nn.GELU(),
            nn.Conv2d(
                128, 128, kernel_size=5, padding=2, groups=64, bias=False
            ),
            nn.BatchNorm2d(128, momentum=0.0061),
=======
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
>>>>>>> REPLACE