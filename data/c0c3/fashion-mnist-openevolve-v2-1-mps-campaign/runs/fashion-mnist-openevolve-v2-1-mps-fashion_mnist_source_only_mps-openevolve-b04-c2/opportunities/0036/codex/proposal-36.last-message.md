MECHANISM: Near-optimal extended-horizon batch-normalization statistics

HYPOTHESIS: Setting batch-normalization momentum to 0.004 will exceed 9,278 correct predictions by modestly extending the successful 0.005 averaging horizon without the severe statistic staleness observed at 0.0025.

INTENDED_EDIT: Set all four batch-normalization layers to momentum 0.004 and retain the best reference design’s 1.17817 evaluation calibration.

EVIDENCE: Momentum 0.005 achieved the best result of 9,278 correct, while 0.01 reached 9,277 and 0.0025 regressed to 9,237; 0.004 is a conservative unexplored point between the best setting and the lower-momentum failure.

<<<<<<< SEARCH
            nn.BatchNorm2d(32),
=======
            nn.BatchNorm2d(32, momentum=0.004),
>>>>>>> REPLACE

<<<<<<< SEARCH
            nn.BatchNorm2d(64),
=======
            nn.BatchNorm2d(64, momentum=0.004),
>>>>>>> REPLACE

<<<<<<< SEARCH
            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
=======
            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128, momentum=0.004),
>>>>>>> REPLACE

<<<<<<< SEARCH
            nn.BatchNorm2d(128),
            nn.GELU(),
            nn.MaxPool2d(2),
=======
            nn.BatchNorm2d(128, momentum=0.004),
            nn.GELU(),
            nn.MaxPool2d(2),
>>>>>>> REPLACE

<<<<<<< SEARCH
            logits = 1.17810 * logits
=======
            logits = 1.17817 * logits
>>>>>>> REPLACE