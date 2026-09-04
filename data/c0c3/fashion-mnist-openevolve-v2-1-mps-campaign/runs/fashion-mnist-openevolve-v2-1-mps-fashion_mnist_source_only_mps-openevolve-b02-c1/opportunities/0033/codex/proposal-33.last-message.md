MECHANISM: Boundary-preserving final pooling

HYPOTHESIS: Retaining a 4×4 final feature map while narrowing the classifier to 80 units will exceed 9,280 correct predictions by preserving spatial boundary information within the parameter ceiling.

INTENDED_EDIT: Enable ceiling mode on the final max pool and resize the classifier for the resulting 4×4 features, yielding 244,442 learned parameters.

EVIDENCE: Learned downsampling timed out, while residual refinements retaining the 3×3 pooling bottleneck fell to 9,266 correct; this tests the spatial bottleneck with negligible convolutional overhead.

<<<<<<< SEARCH
            nn.Conv2d(64, 96, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(96),
            nn.GELU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(96 * 3 * 3, 128),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(128, 10),
        )
=======
            nn.Conv2d(64, 96, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(96),
            nn.GELU(),
            nn.MaxPool2d(2, ceil_mode=True),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(96 * 4 * 4, 80),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(80, 10),
        )
>>>>>>> REPLACE