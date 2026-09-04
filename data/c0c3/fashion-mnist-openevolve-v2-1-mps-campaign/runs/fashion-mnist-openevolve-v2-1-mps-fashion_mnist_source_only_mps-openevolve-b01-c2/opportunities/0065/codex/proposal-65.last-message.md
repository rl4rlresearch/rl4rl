MECHANISM: Dense residual-bottleneck capacity reallocation

HYPOTHESIS: Narrowing the residual refinement’s hidden channels from 64 to 48 while widening the classifier bottleneck from 35 to 40 will exceed 9,330 correct predictions, remain below 250,000 parameters, and reduce computation slightly.

INTENDED_EDIT: Reallocate parameters from the full-width residual branch to the spatial classifier while preserving all verified training, augmentation, scheduling, and inference settings.

EVIDENCE: The current dense architecture and training procedure reliably achieved 9,330 correct; the prior grouped-convolution reallocation was not validated because it timed out, motivating a simpler dense bottleneck that increases classifier capacity without adding operators.

<<<<<<< SEARCH
        self.refine = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
        )
=======
        self.refine = nn.Sequential(
            nn.Conv2d(64, 48, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(48),
            nn.GELU(),
            nn.Conv2d(48, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 35),
            nn.LayerNorm(35),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(35, 10),
        )
=======
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 40),
            nn.LayerNorm(40),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(40, 10),
        )
>>>>>>> REPLACE