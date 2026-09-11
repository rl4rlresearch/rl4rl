MECHANISM: Overlapping phase-stable spatial pooling

HYPOTHESIS: Replacing non-overlapping 2×2 pooling with overlapping 3×3 stride-2 pooling will exceed 9,240 correct predictions by reducing one-pixel pooling-phase sensitivity without adding parameters, while the verified 0.04→0 smoothing schedule preserves the strongest training objective.

INTENDED_EDIT: Use overlapping pooling at both downsampling stages and adopt the best verified full-horizon label-smoothing anneal; retain the proven architecture, optimizer, TTA, and calibration otherwise.

EVIDENCE: The classifier requires translation augmentation and 25-shift evaluation, indicating residual shift sensitivity, while an added spatial refinement block fell to 9,232 correct. This motivates parameter-free downsampling stability instead of more capacity; Reference Design 1 establishes 0.04→0 smoothing as the strongest verified objective at 9,240 correct.

<<<<<<< SEARCH
            nn.GELU(),
            nn.MaxPool2d(2),
            nn.Conv2d(24, 48, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(48),
            nn.GELU(),
            nn.MaxPool2d(2),
            nn.Conv2d(48, 80, kernel_size=3, padding=1, bias=False),
=======
            nn.GELU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.Conv2d(24, 48, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(48),
            nn.GELU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.Conv2d(48, 80, kernel_size=3, padding=1, bias=False),
>>>>>>> REPLACE

<<<<<<< SEARCH
    smoothing = 0.03 + 0.02 * math.cos(math.pi * progress)
=======
    smoothing = 0.02 * (1.0 + math.cos(math.pi * progress))
>>>>>>> REPLACE