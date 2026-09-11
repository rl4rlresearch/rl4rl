MECHANISM: Depthwise-separable spatial refinement with post-filter channel mixing

HYPOTHESIS: Adding pointwise channel mixing after the depthwise convolution while narrowing the dense hidden layer will exceed 9,258 correct predictions without exceeding 250,000 parameters.

INTENDED_EDIT: Insert a batch-normalized 1×1 convolution after depthwise refinement and reduce the classifier width from 128 to 112, preserving all verified training settings and using approximately 241,210 learned parameters.

EVIDENCE: The unaugmented 243,178-parameter design achieved 9,258 correct, while horizontal flips and translations reduced accuracy; this motivates retaining its verified optimization path and reallocating dense-layer capacity to more expressive learned features.

<<<<<<< SEARCH
            nn.Conv2d(
                128, 128, kernel_size=3, padding=1, groups=128, bias=False
            ),
            nn.BatchNorm2d(128),
            nn.GELU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 3 * 3, 128),
            nn.GELU(),
            nn.Dropout(p=0.05),
            nn.Linear(128, 10),
        )
=======
            nn.Conv2d(
                128, 128, kernel_size=3, padding=1, groups=128, bias=False
            ),
            nn.BatchNorm2d(128),
            nn.GELU(),
            nn.Conv2d(128, 128, kernel_size=1, bias=False),
            nn.BatchNorm2d(128),
            nn.GELU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 3 * 3, 112),
            nn.GELU(),
            nn.Dropout(p=0.05),
            nn.Linear(112, 10),
        )
>>>>>>> REPLACE