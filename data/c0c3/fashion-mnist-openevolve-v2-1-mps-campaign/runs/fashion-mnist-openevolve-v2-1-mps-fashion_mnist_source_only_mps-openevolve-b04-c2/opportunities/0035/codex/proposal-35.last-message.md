MECHANISM: Extended-horizon batch-normalization statistics

HYPOTHESIS: Halving all batch-normalization momenta from 0.005 to 0.0025 will exceed 9,278 correct predictions by further stabilizing evaluation statistics without changing training-mode normalization, capacity, or optimization.

INTENDED_EDIT: Set momentum=0.0025 on all four batch-normalization layers while preserving the verified architecture, schedule, loss, and evaluation calibration.

EVIDENCE: Successive momentum reductions improved correctness from 9,273 at 0.1 to 9,276 at 0.02, 9,277 at 0.01, and 9,278 at 0.005; another halving is the most direct test of that isolated trend.

<<<<<<< SEARCH
        self.norm = nn.BatchNorm2d(128)
=======
        self.norm = nn.BatchNorm2d(128, momentum=0.0025)
>>>>>>> REPLACE

<<<<<<< SEARCH
            nn.BatchNorm2d(32),
=======
            nn.BatchNorm2d(32, momentum=0.0025),
>>>>>>> REPLACE

<<<<<<< SEARCH
            nn.BatchNorm2d(64),
=======
            nn.BatchNorm2d(64, momentum=0.0025),
>>>>>>> REPLACE

<<<<<<< SEARCH
            nn.BatchNorm2d(128),
=======
            nn.BatchNorm2d(128, momentum=0.0025),
>>>>>>> REPLACE