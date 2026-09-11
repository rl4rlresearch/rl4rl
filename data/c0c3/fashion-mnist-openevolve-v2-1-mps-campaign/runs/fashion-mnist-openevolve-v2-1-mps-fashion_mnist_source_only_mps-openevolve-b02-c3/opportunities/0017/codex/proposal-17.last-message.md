MECHANISM: Backbone-wide boundary-consistent convolution

HYPOTHESIS: Replicate-padding every 3×3 backbone convolution will exceed 9,273 correct predictions by eliminating artificial zero-border features throughout the network while preserving the qualified architecture and augmentation recipe.

INTENDED_EDIT: Add replicate padding to all four 3×3 feature-extraction convolutions; leave parameters, optimizer, training augmentation, and inference ensemble unchanged.

EVIDENCE: Changing only the 7×7 refinement convolution from zero to replicate padding improved correctness from 9,270 to 9,273 at identical parameter count, while training translations already use replicate padding.

<<<<<<< SEARCH
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
=======
        self.features = nn.Sequential(
            nn.Conv2d(
                1, 32, kernel_size=3, padding=1, bias=False,
                padding_mode="replicate",
            ),
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.Conv2d(
                32, 32, kernel_size=3, padding=1, bias=False,
                padding_mode="replicate",
            ),
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.MaxPool2d(2),
            nn.Conv2d(
                32, 64, kernel_size=3, padding=1, bias=False,
                padding_mode="replicate",
            ),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.Conv2d(
                64, 64, kernel_size=3, padding=1, bias=False,
                padding_mode="replicate",
            ),
            nn.BatchNorm2d(64),
>>>>>>> REPLACE