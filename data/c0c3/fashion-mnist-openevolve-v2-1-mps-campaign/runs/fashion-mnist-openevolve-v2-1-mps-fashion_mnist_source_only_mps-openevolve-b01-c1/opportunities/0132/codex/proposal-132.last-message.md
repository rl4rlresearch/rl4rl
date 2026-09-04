MECHANISM: Compute-efficient feature-channel reallocation

HYPOTHESIS: Widening the residual representation from 56 to 60 channels while narrowing the classifier bottleneck from 64 to 63 will exceed 9,262 correct predictions by investing capacity in spatial feature extraction without the expensive extra block that timed out.

INTENDED_EDIT: Reallocate parameters from the flattened classifier into the existing residual and shortcut paths, retaining the proven topology and training procedure while increasing the model to 248,825 learned parameters.

EVIDENCE: Expanding only the classifier head to 249,976 parameters regressed sharply to 9,221 correct, while adding a second context-gated residual block exceeded the time limit; modestly widening the existing residual block tests richer spatial features with substantially less added computation.

<<<<<<< SEARCH
        self.residual = nn.Sequential(
            nn.Conv2d(40, 56, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(56),
            nn.GELU(),
            nn.Conv2d(56, 56, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(56),
        )
        self.shortcut = nn.Sequential(
            nn.Conv2d(40, 56, kernel_size=1, bias=False),
            nn.BatchNorm2d(56),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(56 * 7 * 7, 64),
            nn.GELU(),
            nn.LayerNorm(64),
            nn.Linear(64, 10),
        )
=======
        self.residual = nn.Sequential(
            nn.Conv2d(40, 60, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(60),
            nn.GELU(),
            nn.Conv2d(60, 60, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(60),
        )
        self.shortcut = nn.Sequential(
            nn.Conv2d(40, 60, kernel_size=1, bias=False),
            nn.BatchNorm2d(60),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(60 * 7 * 7, 63),
            nn.GELU(),
            nn.LayerNorm(63),
            nn.Linear(63, 10),
        )
>>>>>>> REPLACE