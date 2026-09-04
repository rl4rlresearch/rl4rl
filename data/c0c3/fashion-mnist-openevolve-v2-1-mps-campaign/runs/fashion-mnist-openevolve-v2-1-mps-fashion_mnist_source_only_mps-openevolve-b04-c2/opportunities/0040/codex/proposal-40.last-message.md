MECHANISM: Fine-grained batch-normalization averaging horizon

HYPOTHESIS: Setting all batch-normalization momenta to 0.0065 will exceed 9,279 correct predictions by interpolating between the best verified momentum of 0.006 and the nearby 0.007 result.

INTENDED_EDIT: Apply momentum=0.0065 to all four batch-normalization layers and adopt the best reference design’s 1.17817 evaluation calibration.

EVIDENCE: Momentum 0.006 achieved 9,279 correct, while both neighboring tested settings—0.005 and 0.007—achieved 9,278; 0.0065 is the closest unexplored setting on the less failure-prone higher-momentum side.

<<<<<<< SEARCH
            nn.Conv2d(1, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
=======
            nn.Conv2d(1, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32, momentum=0.0065),
>>>>>>> REPLACE

<<<<<<< SEARCH
            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
=======
            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64, momentum=0.0065),
>>>>>>> REPLACE

<<<<<<< SEARCH
            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
=======
            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128, momentum=0.0065),
>>>>>>> REPLACE

<<<<<<< SEARCH
            nn.Conv2d(
                128, 128, kernel_size=5, padding=2, groups=64, bias=False
            ),
            nn.BatchNorm2d(128),
=======
            nn.Conv2d(
                128, 128, kernel_size=5, padding=2, groups=64, bias=False
            ),
            nn.BatchNorm2d(128, momentum=0.0065),
>>>>>>> REPLACE

<<<<<<< SEARCH
        if not self.training:
            logits = 1.17810 * logits
=======
        if not self.training:
            logits = 1.17817 * logits
>>>>>>> REPLACE