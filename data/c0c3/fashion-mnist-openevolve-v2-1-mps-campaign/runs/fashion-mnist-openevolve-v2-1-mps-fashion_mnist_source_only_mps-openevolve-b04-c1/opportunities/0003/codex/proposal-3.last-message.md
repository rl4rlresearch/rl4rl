MECHANISM: Parameter-budgeted two-convolution residual blocks

HYPOTHESIS: Deepening both residual branches while retaining the proven flip ensemble will exceed 9,212 correct validation predictions.

INTENDED_EDIT: Add a second convolution to each residual block and narrow the hidden classifier from 48 to 44 units, yielding 249,934 learned parameters.

EVIDENCE: The wider residual CNN improved validation correct from 8,928 to 9,194, and flip ensembling raised it to 9,212; this motivates investing the remaining parameter budget in additional residual feature extraction while preserving the successful evaluation method.

<<<<<<< SEARCH
        self.residual1 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
        )
=======
        self.residual1 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.residual2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
        )
=======
        self.residual2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
            nn.Linear(64 * 7 * 7, 48),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(48, 10),
=======
            nn.Linear(64 * 7 * 7, 44),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(44, 10),
>>>>>>> REPLACE