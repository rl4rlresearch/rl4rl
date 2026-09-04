MECHANISM: EMA-timescale BatchNorm statistics

HYPOTHESIS: Setting BatchNorm momentum to 0.02 will exceed 9,328 correct predictions by smoothing evaluation statistics over roughly the same 50-step timescale as the parameter EMA.

INTENDED_EDIT: Reduce every BatchNorm layer’s running-statistics momentum from the 0.1 default to 0.02 without adding per-step state processing.

EVIDENCE: EMA-aligned regularization reached 9,328 correct, while two attempts to average BatchNorm buffers with EMA parameters timed out; matching BatchNorm’s update rate to the existing 0.02 parameter interpolation tests the same state-consistency hypothesis at negligible runtime cost.

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