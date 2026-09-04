MECHANISM: Group-factorized residual refinement with a wider spatial classifier

HYPOTHESIS: Reallocating residual-convolution parameters to widen the classifier bottleneck from 35 to 44 will exceed 9,330 correct predictions while remaining under 250,000 parameters.

INTENDED_EDIT: Factor both 64-channel residual convolutions into two-group spatial convolutions followed by pointwise mixing, and spend the saved capacity on the classifier head; all proven training and inference procedures remain unchanged.

EVIDENCE: The current architecture and symmetric logit ensemble reached 9,330 correct, while changes to fusion, smoothing, and EMA were worse; this motivates preserving that procedure and testing a capacity reallocation within the architecture.

<<<<<<< SEARCH
        self.refine = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 35),
            nn.LayerNorm(35),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(35, 10),
        )
=======
        self.refine = nn.Sequential(
            nn.Conv2d(
                64, 64, kernel_size=3, padding=1, groups=2, bias=False
            ),
            nn.Conv2d(64, 64, kernel_size=1, bias=False),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.Conv2d(
                64, 64, kernel_size=3, padding=1, groups=2, bias=False
            ),
            nn.Conv2d(64, 64, kernel_size=1, bias=False),
            nn.BatchNorm2d(64),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 44),
            nn.LayerNorm(44),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(44, 10),
        )
>>>>>>> REPLACE