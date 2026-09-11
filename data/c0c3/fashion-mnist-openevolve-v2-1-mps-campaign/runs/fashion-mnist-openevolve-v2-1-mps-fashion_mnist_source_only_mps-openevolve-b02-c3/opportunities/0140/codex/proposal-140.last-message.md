MECHANISM: Wide factorized spatial-template classifier

HYPOTHESIS: Reallocating parameters from the 56-unit dense head into deeper 97-channel features and 704 learned channel–spatial templates will exceed 9,348 correct predictions by learning richer part detectors while retaining full 7×7 layout.

INTENDED_EDIT: Replace the load-bearing assumption that image layout requires an unrestricted flattening matrix with a wider separable mechanism: pointwise semantic projection followed by learned per-channel 7×7 templates. Add a high-level convolution and widen spatial refinement; expected parameter count is 249,793.

EVIDENCE: Global pooling fell to 9,085 correct because it erased layout, while translation TTA raised correctness to 9,348, showing that spatial organization and alignment matter. The proposed head preserves all 7×7 positions but trades the narrow 56-unit dense bottleneck for 704 nonlinear spatial templates and stronger high-level features.

<<<<<<< SEARCH
            nn.MaxPool2d(2),
            SpatialRefinement(64),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 56),
            nn.GELU(),
            nn.Dropout(p=0.10),
            nn.Linear(56, 10),
        )
=======
            nn.MaxPool2d(2),
            nn.Conv2d(64, 97, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(97),
            nn.GELU(),
            SpatialRefinement(97),
        )
        self.classifier = nn.Sequential(
            nn.Conv2d(97, 704, kernel_size=1, bias=False),
            nn.BatchNorm2d(704),
            nn.GELU(),
            nn.Conv2d(
                704,
                704,
                kernel_size=7,
                groups=704,
            ),
            nn.GELU(),
            nn.Flatten(),
            nn.Dropout(p=0.10),
            nn.Linear(704, 10),
        )
>>>>>>> REPLACE