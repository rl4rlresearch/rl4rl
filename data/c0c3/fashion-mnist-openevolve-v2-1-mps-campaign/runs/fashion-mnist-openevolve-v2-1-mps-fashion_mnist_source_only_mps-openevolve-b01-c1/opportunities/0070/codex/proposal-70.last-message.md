MECHANISM: Learned phase-preserving space-to-depth stem

HYPOTHESIS: Replacing both max-pooling operations with space-to-depth rearrangement and learned channel mixing will exceed 9,247 correct predictions by retaining fine spatial phase information before hierarchical feature extraction.

INTENDED_EDIT: Replace the 16/32-channel max-pooling stem with a 20/40-channel stem whose lossless pixel-unshuffle stages learn how to combine each 2×2 neighborhood; the resulting 32×7×7 representation preserves the proven residual decoder and training objective at 239,918 parameters.

EVIDENCE: The localized raw-image shortcut reached only 9,224 correct and added post-pooling residual refinement reached 9,204, indicating that neither restoring low-level evidence late nor further processing already-pooled features repairs the representation. This challenges the shared assumption that fixed max-pooling should discard spatial phase early and instead learns the downsampling computation before information is lost.

<<<<<<< SEARCH
        self.stem = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.GELU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.MaxPool2d(2),
        )
=======
        self.stem = nn.Sequential(
            nn.Conv2d(1, 20, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(20),
            nn.GELU(),
            nn.PixelUnshuffle(2),
            nn.Conv2d(80, 20, kernel_size=1, bias=False),
            nn.BatchNorm2d(20),
            nn.GELU(),
            nn.Conv2d(20, 40, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(40),
            nn.GELU(),
            nn.PixelUnshuffle(2),
            nn.Conv2d(160, 32, kernel_size=1, bias=False),
            nn.BatchNorm2d(32),
            nn.GELU(),
        )
>>>>>>> REPLACE