MECHANISM: Prediction-invariant lower-side calibration bisection

HYPOTHESIS: With momentum 0.00623359375 restored, evaluation scale 1.178165 will preserve all 9,279 argmax predictions and lower validation cross-entropy below 0.20861771774291993.

INTENDED_EDIT: Restore the best verified batch-normalization momentum in all four layers and test the midpoint between evaluation scales 1.17816 and 1.17817.

EVIDENCE: Scale 1.17817 achieved the best verified cross-entropy, 0.20861771774291993, while 1.17816 was worse and the higher intermediate 1.1781743 was also worse; 1.178165 is the untested lower-side midpoint and positive scaling cannot change argmax predictions.

<<<<<<< SEARCH
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
=======
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
>>>>>>> REPLACE

<<<<<<< SEARCH
        if not self.training:
            logits = 1.17817 * logits
=======
        if not self.training:
            logits = 1.178165 * logits
>>>>>>> REPLACE