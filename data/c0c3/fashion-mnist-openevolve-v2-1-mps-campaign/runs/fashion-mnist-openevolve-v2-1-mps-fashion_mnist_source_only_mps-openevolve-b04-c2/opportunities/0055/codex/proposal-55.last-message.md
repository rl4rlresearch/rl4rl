MECHANISM: Accuracy-boundary batch-normalization bisection

HYPOTHESIS: Momentum 0.006233642578125 will retain 9,279 correct predictions while lowering validation cross-entropy below 0.208617718.

INTENDED_EDIT: Set all four batch-normalization momenta to the midpoint between the highest successful setting and the nearest one-error setting.

EVIDENCE: Momentum 0.00623359375 achieved 9,279 correct, while 0.00623369140625 lost one prediction despite lower cross-entropy; their midpoint is the most informative remaining boundary probe.

<<<<<<< SEARCH
        self.norm = nn.BatchNorm2d(128, momentum=0.00623125)
=======
        self.norm = nn.BatchNorm2d(128, momentum=0.006233642578125)
>>>>>>> REPLACE

<<<<<<< SEARCH
            nn.Conv2d(1, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32, momentum=0.00623125),
            nn.GELU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64, momentum=0.00623125),
            nn.GELU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128, momentum=0.00623125),
=======
            nn.Conv2d(1, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32, momentum=0.006233642578125),
            nn.GELU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64, momentum=0.006233642578125),
            nn.GELU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128, momentum=0.006233642578125),
>>>>>>> REPLACE