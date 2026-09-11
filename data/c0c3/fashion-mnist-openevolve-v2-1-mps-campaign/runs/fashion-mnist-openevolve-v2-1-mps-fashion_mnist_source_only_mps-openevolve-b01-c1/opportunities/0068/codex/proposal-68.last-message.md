MECHANISM: Low-variance BatchNorm statistic tracking

HYPOTHESIS: Reducing BatchNorm momentum from 0.1 to 0.03 will exceed 9,247 correct predictions by averaging running statistics across more late-training batches, improving fixed validation inference without altering learned capacity or supervision.

INTENDED_EDIT: Set every BatchNorm layer’s running-statistics momentum to 0.03 while preserving the proven architecture, augmentation, loss curricula, optimizer, and evaluation ensemble.

EVIDENCE: The 9,247-correct design remains strongest while added architecture, geometric augmentation, cutout, and parameter averaging all regressed; isolating inference-statistic estimation tests a low-cost stability axis without disturbing the successful computation.

<<<<<<< SEARCH
BATCH_SIZE = 128
GRAD_CLIP_NORM = 5.0
=======
BATCH_SIZE = 128
GRAD_CLIP_NORM = 5.0
BN_MOMENTUM = 0.03
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.stem = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.GELU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.MaxPool2d(2),
        )
=======
        self.stem = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(16, momentum=BN_MOMENTUM),
            nn.GELU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32, momentum=BN_MOMENTUM),
            nn.GELU(),
            nn.MaxPool2d(2),
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.residual = nn.Sequential(
            nn.Conv2d(32, 48, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(48),
            nn.GELU(),
            nn.Conv2d(48, 48, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(48),
        )
        self.shortcut = nn.Sequential(
            nn.Conv2d(32, 48, kernel_size=1, bias=False),
            nn.BatchNorm2d(48),
        )
=======
        self.residual = nn.Sequential(
            nn.Conv2d(32, 48, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(48, momentum=BN_MOMENTUM),
            nn.GELU(),
            nn.Conv2d(48, 48, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(48, momentum=BN_MOMENTUM),
        )
        self.shortcut = nn.Sequential(
            nn.Conv2d(32, 48, kernel_size=1, bias=False),
            nn.BatchNorm2d(48, momentum=BN_MOMENTUM),
        )
>>>>>>> REPLACE