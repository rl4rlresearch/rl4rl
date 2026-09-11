MECHANISM: EMA-matched BatchNorm statistic horizon

HYPOTHESIS: Matching BatchNorm momentum to the 0.02 parameter-EMA update rate will exceed 9,328 correct predictions by aligning validation-time running statistics with the averaged weights.

INTENDED_EDIT: Set all four BatchNorm layers’ momentum from the default 0.1 to 0.02 without changing parameters or computation.

EVIDENCE: Live paired-view BatchNorm tied the best accuracy at 9,328 but slightly worsened cross-entropy, showing normalization-state handling affects the result; the current final buffers use a much shorter averaging horizon than the 0.02 EMA parameters.

<<<<<<< SEARCH
        self.stem = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.GELU(),
        )
        self.residual1 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
        )
=======
        self.stem = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32, momentum=0.02),
            nn.GELU(),
        )
        self.residual1 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32, momentum=0.02),
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.transition = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.GELU(),
        )
        self.residual2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
        )
=======
        self.transition = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64, momentum=0.02),
            nn.GELU(),
        )
        self.residual2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64, momentum=0.02),
        )
>>>>>>> REPLACE