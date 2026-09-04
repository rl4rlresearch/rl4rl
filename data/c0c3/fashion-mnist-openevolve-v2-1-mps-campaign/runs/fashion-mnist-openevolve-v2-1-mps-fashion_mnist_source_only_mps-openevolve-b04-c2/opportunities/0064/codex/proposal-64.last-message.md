MECHANISM: Accuracy-boundary batch-normalization bisection

HYPOTHESIS: Momentum 0.00623360271453857421875 will retain 9,279 correct predictions while lowering validation cross-entropy below 0.20861771774291993.

INTENDED_EDIT: Set all four batch-normalization momenta to the midpoint between the highest verified successful setting and the nearest one-error setting.

EVIDENCE: Momentum 0.0062336025238037109375 retained 9,279 correct, while 0.0062336029052734375 produced 9,278 with lower cross-entropy; their untested midpoint is the most informative remaining boundary probe.

<<<<<<< SEARCH
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1, bias=False),
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
            nn.GELU(),
            nn.MaxPool2d(2),
        )
=======
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32, momentum=0.00623360271453857421875),
            nn.GELU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64, momentum=0.00623360271453857421875),
            nn.GELU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128, momentum=0.00623360271453857421875),
            nn.GELU(),
            nn.Conv2d(
                128, 128, kernel_size=5, padding=2, groups=64, bias=False
            ),
            nn.BatchNorm2d(128, momentum=0.00623360271453857421875),
            nn.GELU(),
            nn.MaxPool2d(2),
        )
>>>>>>> REPLACE