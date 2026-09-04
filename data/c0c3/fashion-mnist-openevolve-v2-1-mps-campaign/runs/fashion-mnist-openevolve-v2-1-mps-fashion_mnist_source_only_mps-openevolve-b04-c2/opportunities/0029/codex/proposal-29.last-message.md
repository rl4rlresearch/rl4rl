MECHANISM: Longer-horizon batch-normalization statistics

HYPOTHESIS: Reducing all batch-normalization momenta from 0.02 to 0.01 will exceed 9,276 correct predictions by further reducing noise in evaluation statistics without changing training-mode normalization, capacity, or optimization.

INTENDED_EDIT: Set every batch-normalization layer’s momentum to 0.01 while preserving the verified architecture, training procedure, and evaluation calibration.

EVIDENCE: Lowering momentum from the default 0.1 to 0.02 improved correctness from 9,273 to 9,276; extending that isolated trend is the most informative next test after other architecture, augmentation, loss, and test-time ensemble changes regressed.

<<<<<<< SEARCH
            nn.Conv2d(1, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32, momentum=0.02),
            nn.GELU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64, momentum=0.02),
            nn.GELU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128, momentum=0.02),
            nn.GELU(),
            nn.Conv2d(
                128, 128, kernel_size=5, padding=2, groups=64, bias=False
            ),
            nn.BatchNorm2d(128, momentum=0.02),
=======
            nn.Conv2d(1, 32, kernel_size=3, padding=1, bias=False),
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
>>>>>>> REPLACE