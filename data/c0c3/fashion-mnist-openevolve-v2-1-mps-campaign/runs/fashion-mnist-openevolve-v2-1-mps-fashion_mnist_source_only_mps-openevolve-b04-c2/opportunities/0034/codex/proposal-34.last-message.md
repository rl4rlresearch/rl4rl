MECHANISM: Extended-horizon batch-normalization statistics

HYPOTHESIS: Reducing all four batch-normalization momenta from 0.01 to 0.005 will exceed 9,277 correct predictions by stabilizing evaluation statistics without altering training-mode normalization, optimization, or capacity.

INTENDED_EDIT: Set every batch-normalization layer’s momentum to 0.005 while preserving the verified architecture, loss, schedule, and evaluation calibration.

EVIDENCE: The otherwise equivalent reference design with momentum 0.005 achieved 9,278 correct, improving on 9,277 at momentum 0.01 and continuing the observed gains from progressively lower momentum.

<<<<<<< SEARCH
        self.norm = nn.BatchNorm2d(128, momentum=0.01)
=======
        self.norm = nn.BatchNorm2d(128, momentum=0.005)
>>>>>>> REPLACE

<<<<<<< SEARCH
            nn.BatchNorm2d(32, momentum=0.01),
=======
            nn.BatchNorm2d(32, momentum=0.005),
>>>>>>> REPLACE

<<<<<<< SEARCH
            nn.BatchNorm2d(64, momentum=0.01),
=======
            nn.BatchNorm2d(64, momentum=0.005),
>>>>>>> REPLACE

<<<<<<< SEARCH
            nn.BatchNorm2d(128, momentum=0.01),
=======
            nn.BatchNorm2d(128, momentum=0.005),
>>>>>>> REPLACE