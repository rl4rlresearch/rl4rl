MECHANISM: Tail-aligned BatchNorm statistics

HYPOTHESIS: Reducing BatchNorm momentum to 0.01 will better align evaluation statistics with the tail-averaged parameters and achieve at least 9,312 correct predictions without increasing parameters or computation.

INTENDED_EDIT: Change every convolutional and classifier BatchNorm layer from the default 0.1 momentum to 0.01; leave architecture, optimization, augmentation, averaging, and TTA unchanged.

EVIDENCE: Tail parameter averaging at weight 0.5 produced 9,311 correct, while reducing its weight regressed to 9,307; explicit buffer-averaging attempts timed out, motivating a computation-neutral way to make running statistics represent a longer, tail-aligned training window.

<<<<<<< SEARCH
            nn.Conv2d(1, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 96, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(96),
=======
            nn.Conv2d(1, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32, momentum=0.01),
            nn.GELU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32, momentum=0.01),
            nn.GELU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64, momentum=0.01),
            nn.GELU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64, momentum=0.01),
            nn.GELU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 96, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(96, momentum=0.01),
>>>>>>> REPLACE

<<<<<<< SEARCH
            nn.Linear(96 * 3 * 3, 147),
            nn.BatchNorm1d(147),
            nn.GELU(),
=======
            nn.Linear(96 * 3 * 3, 147),
            nn.BatchNorm1d(147, momentum=0.01),
            nn.GELU(),
>>>>>>> REPLACE