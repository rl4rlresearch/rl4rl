MECHANISM: Increased dilated-context channel allocation

HYPOTHESIS: Reallocating eight aggregator channels from the local branch to the successful dilation-2 context branch will exceed 9,265 correct predictions by strengthening broader garment-shape modeling while retaining 40 channels for local detail and keeping parameter count unchanged.

INTENDED_EDIT: Change the late aggregator split from 48 local and 16 dilated-context channels to 40 local and 24 dilated-context channels; preserve the 64-channel output, classifier width, training procedure, TTA, and calibration.

EVIDENCE: Replacing the 64-channel local aggregator, which achieved 9,255 correct, with a 48-local/16-context split improved correctness to 9,265 at the identical 245,150-parameter count. This motivates a controlled dose increase of the only architectural addition shown to improve the current best design.

<<<<<<< SEARCH
        self.aggregate_local = nn.Sequential(
            nn.Conv2d(
                80, 48, kernel_size=3, stride=2, padding=1, bias=False
            ),
            nn.BatchNorm2d(48),
            nn.GELU(),
        )
        self.aggregate_context = nn.Sequential(
            nn.Conv2d(
                80,
                16,
                kernel_size=3,
                stride=2,
                padding=2,
                dilation=2,
                bias=False,
            ),
            nn.BatchNorm2d(16),
            nn.GELU(),
        )
=======
        self.aggregate_local = nn.Sequential(
            nn.Conv2d(
                80, 40, kernel_size=3, stride=2, padding=1, bias=False
            ),
            nn.BatchNorm2d(40),
            nn.GELU(),
        )
        self.aggregate_context = nn.Sequential(
            nn.Conv2d(
                80,
                24,
                kernel_size=3,
                stride=2,
                padding=2,
                dilation=2,
                bias=False,
            ),
            nn.BatchNorm2d(24),
            nn.GELU(),
        )
>>>>>>> REPLACE