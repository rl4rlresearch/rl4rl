MECHANISM: Longer-horizon batch-normalization statistics

HYPOTHESIS: Setting all four batch-normalization momenta to 0.01 will reproduce the verified 9,277-correct result, improving over the current 9,273 correct predictions without changing training optimization or capacity.

INTENDED_EDIT: Add momentum=0.01 to every batch-normalization layer while preserving the architecture, loss, schedule, and evaluation calibration.

EVIDENCE: Reference Design 1 is otherwise identical and achieved 9,277 correct with momentum 0.01, compared with 9,273 for the current default-momentum design.

<<<<<<< SEARCH
        self.norm = nn.BatchNorm2d(128)
=======
        self.norm = nn.BatchNorm2d(128, momentum=0.01)
>>>>>>> REPLACE

<<<<<<< SEARCH
            nn.BatchNorm2d(32),
=======
            nn.BatchNorm2d(32, momentum=0.01),
>>>>>>> REPLACE

<<<<<<< SEARCH
            nn.BatchNorm2d(64),
=======
            nn.BatchNorm2d(64, momentum=0.01),
>>>>>>> REPLACE

<<<<<<< SEARCH
            nn.BatchNorm2d(128),
            nn.GELU(),
            PairwiseRefinement(),
=======
            nn.BatchNorm2d(128, momentum=0.01),
            nn.GELU(),
            PairwiseRefinement(),
>>>>>>> REPLACE