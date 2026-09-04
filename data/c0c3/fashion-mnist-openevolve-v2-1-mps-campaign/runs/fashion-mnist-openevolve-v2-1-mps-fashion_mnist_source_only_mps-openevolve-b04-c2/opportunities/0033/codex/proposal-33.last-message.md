MECHANISM: Extended-horizon batch-normalization statistics

HYPOTHESIS: Reducing all batch-normalization momenta from 0.01 to 0.005 will exceed 9,277 correct predictions by further stabilizing evaluation statistics without altering training-mode normalization, optimization, or capacity.

INTENDED_EDIT: Set every batch-normalization layer’s momentum to 0.005 while preserving the verified architecture, schedule, loss, and evaluation calibration.

EVIDENCE: Lowering momentum from the default 0.1 to 0.02 improved correctness from 9,273 to 9,276, and lowering it again to 0.01 improved correctness to 9,277; testing 0.005 is the most direct continuation of that isolated trend.

<<<<<<< SEARCH
            nn.BatchNorm2d(32, momentum=0.01),
            nn.GELU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64, momentum=0.01),
            nn.GELU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128, momentum=0.01),
            nn.GELU(),
            nn.Conv2d(
                128, 128, kernel_size=5, padding=2, groups=64, bias=False
            ),
            nn.BatchNorm2d(128, momentum=0.01),
=======
            nn.BatchNorm2d(32, momentum=0.005),
            nn.GELU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64, momentum=0.005),
            nn.GELU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128, momentum=0.005),
            nn.GELU(),
            nn.Conv2d(
                128, 128, kernel_size=5, padding=2, groups=64, bias=False
            ),
            nn.BatchNorm2d(128, momentum=0.005),
>>>>>>> REPLACE