MECHANISM: Longer-horizon batch-normalization statistics

HYPOTHESIS: Reducing all four batch-normalization momenta from 0.02 to 0.01 will reproduce the verified 9,277-correct result and exceed the current 9,276 correct predictions without changing optimization or capacity.

INTENDED_EDIT: Set every batch-normalization layer’s momentum to 0.01 while preserving the architecture, training schedule, loss, and evaluation calibration.

EVIDENCE: Reference Design 1 made this isolated momentum change on the same 248,426-parameter procedure and achieved 9,277 correct versus 9,276 at momentum 0.02.

<<<<<<< SEARCH
        self.norm = nn.BatchNorm2d(128, momentum=0.02)
=======
        self.norm = nn.BatchNorm2d(128, momentum=0.01)
>>>>>>> REPLACE

<<<<<<< SEARCH
            nn.BatchNorm2d(32, momentum=0.02),
=======
            nn.BatchNorm2d(32, momentum=0.01),
>>>>>>> REPLACE

<<<<<<< SEARCH
            nn.BatchNorm2d(64, momentum=0.02),
=======
            nn.BatchNorm2d(64, momentum=0.01),
>>>>>>> REPLACE

<<<<<<< SEARCH
            nn.BatchNorm2d(128, momentum=0.02),
=======
            nn.BatchNorm2d(128, momentum=0.01),
>>>>>>> REPLACE