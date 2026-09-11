MECHANISM: Boundary-covering adaptive final pooling

HYPOTHESIS: Covering the entire 7×7 feature map while retaining a 3×3 classifier input will exceed 9,280 correct predictions by preserving boundary features currently discarded by floor-mode max pooling.

INTENDED_EDIT: Replace only the final 2×2 max pool with adaptive 3×3 max pooling, preserving the parameter count, classifier, training procedure, and output shape.

EVIDENCE: The 4×4 ceiling-mode experiment targeted the same boundary-loss issue but timed out with an expanded classifier; adaptive pooling tests boundary coverage without its parameter or runtime overhead.

<<<<<<< SEARCH
            nn.Conv2d(64, 96, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(96),
            nn.GELU(),
            nn.MaxPool2d(2),
        )
=======
            nn.Conv2d(64, 96, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(96),
            nn.GELU(),
            nn.AdaptiveMaxPool2d((3, 3)),
        )
>>>>>>> REPLACE