MECHANISM: Learned polyphase downsampling

HYPOTHESIS: Replacing both information-discarding max pools with space-to-depth phase preservation and learned channel mixing will exceed 9,268 correct predictions by retaining subpixel garment structure for the convolutional classifier.

INTENDED_EDIT: Replace the shared max-pooling backbone with PixelUnshuffle-based learned polyphase reductions while retaining the successful spatial classifier, and restore the best verified vertical-biased TTA weights.

EVIDENCE: Head replacement and global channel gating regressed to 9,227 and 9,257 correct while retaining the same max-pooled representation; meanwhile, prediction quality responded to one-pixel TTA orientation, motivating a clean test of whether preserving sampling-phase information before learned mixing improves the representation.

<<<<<<< SEARCH
        self.features = nn.Sequential(
            nn.Conv2d(5, 24, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(24),
            nn.GELU(),
            nn.MaxPool2d(2),
            nn.Conv2d(24, 48, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(48),
            nn.GELU(),
            nn.MaxPool2d(2),
            nn.Conv2d(48, 80, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(80),
            nn.GELU(),
        )
=======
        self.features = nn.Sequential(
            nn.Conv2d(5, 24, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(24),
            nn.GELU(),
            nn.PixelUnshuffle(2),
            nn.Conv2d(96, 48, kernel_size=1, bias=False),
            nn.BatchNorm2d(48),
            nn.GELU(),
            nn.Conv2d(48, 48, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(48),
            nn.GELU(),
            nn.PixelUnshuffle(2),
            nn.Conv2d(192, 80, kernel_size=1, bias=False),
            nn.BatchNorm2d(80),
            nn.GELU(),
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
        logits = 0.38 * self._flip_average(images)

        for delta_y in range(-2, 3):
            for delta_x in range(-2, 3):
                if delta_y == 0 and delta_x == 0:
                    continue

                shifted = padded[
                    :,
                    :,
                    2 + delta_y : 2 + delta_y + height,
                    2 + delta_x : 2 + delta_x + width,
                ]
                radius = max(abs(delta_y), abs(delta_x))
                if radius == 1:
                    weight = (
                        0.09
                        if delta_y == 0 or delta_x == 0
                        else 0.04
                    )
=======
        logits = 0.36 * self._flip_average(images)

        for delta_y in range(-2, 3):
            for delta_x in range(-2, 3):
                if delta_y == 0 and delta_x == 0:
                    continue

                shifted = padded[
                    :,
                    :,
                    2 + delta_y : 2 + delta_y + height,
                    2 + delta_x : 2 + delta_x + width,
                ]
                radius = max(abs(delta_y), abs(delta_x))
                if radius == 1:
                    if delta_x == 0:
                        weight = 0.10
                    elif delta_y == 0:
                        weight = 0.08
                    else:
                        weight = 0.045
>>>>>>> REPLACE