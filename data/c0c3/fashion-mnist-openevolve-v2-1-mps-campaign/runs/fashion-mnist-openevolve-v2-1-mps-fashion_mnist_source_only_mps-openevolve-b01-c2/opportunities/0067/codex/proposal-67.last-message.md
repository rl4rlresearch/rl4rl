MECHANISM: Compute-efficient residual bottleneck capacity reallocation

HYPOTHESIS: Replacing the dense refinement with a 1×1–3×3–1×1 bottleneck and widening the classifier to 52 units will exceed 9,330 correct predictions while reducing convolutional work enough to finish verification, with 248,750 learned parameters.

INTENDED_EDIT: Compress the residual refinement through 40 channels, add an intermediate nonlinearity, and reallocate the saved parameters to the spatial classifier.

EVIDENCE: The verified model reached 9,330 correct but required 80.73 seconds, while many equally or more expensive variants timed out; the 48-channel/40-unit reallocation also timed out, motivating a substantially cheaper bottleneck that preserves spatial refinement and supports a wider classifier.

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
            nn.Conv2d(64, 40, kernel_size=1, bias=False),
            nn.BatchNorm2d(40),
            nn.GELU(),
            nn.Conv2d(40, 40, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(40),
            nn.GELU(),
            nn.Conv2d(40, 64, kernel_size=1, bias=False),
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
            nn.Linear(64 * 7 * 7, 52),
            nn.LayerNorm(52),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(52, 10),
        )
>>>>>>> REPLACE