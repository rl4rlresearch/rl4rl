MECHANISM: Border-complete adaptive spatial pooling

HYPOTHESIS: Replacing the asymmetric final 7×7-to-3×3 max pool with adaptive max pooling will exceed 9,318 correct predictions by retaining bottom/right-border features and improving translation consistency without added parameters or meaningful runtime cost.

INTENDED_EDIT: Change only the final pooling layer from fixed 2×2 pooling, which ignores the last feature-map row and column, to 3×3 adaptive max pooling that covers the entire 7×7 map.

EVIDENCE: The successful 0.03-label-smoothing design reached 9,318 correct with translation augmentation and TTA, while residual refinement reduced correctness; this motivates preserving that learned pipeline and addressing the final encoder’s spatial asymmetry directly.

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