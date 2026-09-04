MECHANISM: Conservative batch-normalization horizon refinement

HYPOTHESIS: Raising momentum from 0.006 to 0.0061 will retain 9,279 correct predictions while slightly lowering cross-entropy, thereby improving validation_score.

INTENDED_EDIT: Set all four batch-normalization momenta to 0.0061 while preserving architecture, optimization, loss, and calibration.

EVIDENCE: Momentum 0.006 produced the best correctness, while 0.0065 produced lower cross-entropy but lost one correct prediction; a smaller upward step tests whether the cross-entropy benefit begins before that accuracy boundary.

<<<<<<< SEARCH
            nn.Conv2d(1, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32, momentum=0.006),
=======
            nn.Conv2d(1, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32, momentum=0.0061),
>>>>>>> REPLACE

<<<<<<< SEARCH
            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64, momentum=0.006),
=======
            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64, momentum=0.0061),
>>>>>>> REPLACE

<<<<<<< SEARCH
            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128, momentum=0.006),
=======
            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128, momentum=0.0061),
>>>>>>> REPLACE

<<<<<<< SEARCH
            nn.Conv2d(
                128, 128, kernel_size=5, padding=2, groups=64, bias=False
            ),
            nn.BatchNorm2d(128, momentum=0.006),
=======
            nn.Conv2d(
                128, 128, kernel_size=5, padding=2, groups=64, bias=False
            ),
            nn.BatchNorm2d(128, momentum=0.0061),
>>>>>>> REPLACE