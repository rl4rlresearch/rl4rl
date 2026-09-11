MECHANISM: EMA-aligned BatchNorm statistics

HYPOTHESIS: Reducing BatchNorm momentum to 0.02 will exceed 9,328 correct predictions by aligning the running-statistics horizon with the successful 0.02 parameter EMA and reducing terminal normalization noise.

INTENDED_EDIT: Set all four BatchNorm layers to momentum 0.02 without changing architecture, learned parameters, loss, or runtime complexity.

EVIDENCE: The 9,328 baseline copies current BatchNorm buffers into the parameter-EMA model, so those statistics retain a much shorter effective horizon than its weights; architecture, fusion, augmentation, and late-loss changes failed to improve the baseline, motivating this isolated stabilization change.

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
        self.stem = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32, momentum=0.02),
            nn.GELU(),
        )
        self.residual1 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32, momentum=0.02),
        )
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