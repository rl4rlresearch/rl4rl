MECHANISM: Near-optimal batch-normalization averaging horizon

HYPOTHESIS: Increasing all batch-normalization momenta from 0.005 to 0.006 will exceed 9,278 correct predictions by reducing statistic staleness while retaining most of the noise suppression responsible for the current best result.

INTENDED_EDIT: Set all four batch-normalization layers to momentum 0.006; preserve the architecture, optimizer, schedule, loss, and evaluation calibration.

EVIDENCE: Momentum 0.005 achieved 9,278 correct, while 0.01 achieved 9,277; reductions below 0.005 regressed sharply to 9,269 at 0.004 and 9,237 at 0.0025. Testing 0.006 explores the safer, higher-momentum side of the observed optimum.

<<<<<<< SEARCH
            nn.Conv2d(1, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32, momentum=0.005),
=======
            nn.Conv2d(1, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32, momentum=0.006),
>>>>>>> REPLACE

<<<<<<< SEARCH
            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64, momentum=0.005),
=======
            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64, momentum=0.006),
>>>>>>> REPLACE

<<<<<<< SEARCH
            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128, momentum=0.005),
=======
            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128, momentum=0.006),
>>>>>>> REPLACE

<<<<<<< SEARCH
            nn.Conv2d(
                128, 128, kernel_size=5, padding=2, groups=64, bias=False
            ),
            nn.BatchNorm2d(128, momentum=0.005),
=======
            nn.Conv2d(
                128, 128, kernel_size=5, padding=2, groups=64, bias=False
            ),
            nn.BatchNorm2d(128, momentum=0.006),
>>>>>>> REPLACE