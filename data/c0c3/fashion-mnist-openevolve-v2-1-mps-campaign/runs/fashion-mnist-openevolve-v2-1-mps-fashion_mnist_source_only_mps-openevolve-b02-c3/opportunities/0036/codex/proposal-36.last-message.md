MECHANISM: Qualified max-pooling backbone with calibrated posterior ensembling

HYPOTHESIS: Restoring Reference Design 2’s fixed max-pooling architecture and temperature 0.75 will recover 9,315 correct predictions with validation cross-entropy near 0.19136.

INTENDED_EDIT: Replace both pixel-unshuffle projections with the verified 32/64-channel max-pooling backbone and sharpen the inference ensemble from temperature 0.78 to 0.75.

EVIDENCE: The current learned-downsampling design achieved 9,291 correct, while Reference Design 2’s otherwise matching max-pooling implementation achieved 9,315 correct and the best verified cross-entropy of 0.1913601.

<<<<<<< SEARCH
        self.features = nn.Sequential(
            nn.Conv2d(1, 24, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(24),
            nn.GELU(),
            nn.Conv2d(24, 24, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(24),
            nn.GELU(),
            nn.PixelUnshuffle(2),
            nn.Conv2d(24 * 4, 48, kernel_size=1, bias=False),
            nn.BatchNorm2d(48),
            nn.GELU(),
            nn.Conv2d(48, 48, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(48),
            nn.GELU(),
            nn.Conv2d(48, 48, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(48),
            nn.GELU(),
            nn.PixelUnshuffle(2),
            nn.Conv2d(48 * 4, 64, kernel_size=1, bias=False),
            nn.BatchNorm2d(64),
            nn.GELU(),
            SpatialRefinement(64),
        )
=======
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.MaxPool2d(2),
            SpatialRefinement(64),
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
        ).log() / 0.78
=======
        ).log() / 0.75
>>>>>>> REPLACE