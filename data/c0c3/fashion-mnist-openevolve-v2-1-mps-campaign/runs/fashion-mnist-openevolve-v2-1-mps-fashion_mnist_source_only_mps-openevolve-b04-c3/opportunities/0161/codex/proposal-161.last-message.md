MECHANISM: High-resolution residual refinement

HYPOTHESIS: A second 48-channel residual depthwise block will exceed 9,251 correct predictions by improving aligned 14×14 feature extraction while remaining below the parameter and runtime limits.

INTENDED_EDIT: Add one stride-preserving residual block before the second max pool, increasing learned parameters from 245,040 to 247,968 while retaining the verified optimizer, augmentation, EMA ensemble, and calibration.

EVIDENCE: Translation augmentation reduced correctness to 9,228, favoring alignment-preserving processing; unlike the heavier architectural experiments that timed out, this block adds only 2,928 parameters and roughly one existing mid-resolution block of computation.

<<<<<<< SEARCH
            nn.GELU(),
            ResidualDepthwiseBlock(48),
            nn.MaxPool2d(2),
            nn.Conv2d(48, 96, kernel_size=3, padding=1, bias=False),
=======
            nn.GELU(),
            ResidualDepthwiseBlock(48),
            ResidualDepthwiseBlock(48),
            nn.MaxPool2d(2),
            nn.Conv2d(48, 96, kernel_size=3, padding=1, bias=False),
>>>>>>> REPLACE