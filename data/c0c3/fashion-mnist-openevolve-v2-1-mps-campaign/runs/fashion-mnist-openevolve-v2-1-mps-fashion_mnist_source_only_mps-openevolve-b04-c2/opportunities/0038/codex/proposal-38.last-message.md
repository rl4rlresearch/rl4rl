MECHANISM: Intermediate-horizon batch-normalization statistics

HYPOTHESIS: Setting all four batch-normalization momenta to 0.007 will exceed 9,279 correct predictions by reducing statistic staleness relative to 0.006 while retaining substantially more smoothing than the weaker 0.01 setting.

INTENDED_EDIT: Increase every batch-normalization momentum from 0.005 to 0.007 while preserving the architecture, optimization, loss, schedule, and evaluation calibration.

EVIDENCE: Momentum 0.006 achieved the best result at 9,279 correct, improving over 9,278 at 0.005, while 0.01 achieved 9,277; 0.007 is the closest unexplored point in the supported higher-momentum direction.

<<<<<<< SEARCH
        self.norm = nn.BatchNorm2d(128, momentum=0.005)
=======
        self.norm = nn.BatchNorm2d(128, momentum=0.007)
>>>>>>> REPLACE

<<<<<<< SEARCH
            nn.BatchNorm2d(32, momentum=0.005),
=======
            nn.BatchNorm2d(32, momentum=0.007),
>>>>>>> REPLACE

<<<<<<< SEARCH
            nn.BatchNorm2d(64, momentum=0.005),
=======
            nn.BatchNorm2d(64, momentum=0.007),
>>>>>>> REPLACE

<<<<<<< SEARCH
            nn.BatchNorm2d(128, momentum=0.005),
=======
            nn.BatchNorm2d(128, momentum=0.007),
>>>>>>> REPLACE