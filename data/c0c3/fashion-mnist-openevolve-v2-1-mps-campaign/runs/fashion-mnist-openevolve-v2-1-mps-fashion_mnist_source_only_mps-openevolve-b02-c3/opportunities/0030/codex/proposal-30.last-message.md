MECHANISM: Learned polyphase-preserving downsampling

HYPOTHESIS: Replacing both hard max-pooling operations with pixel-unshuffle plus learned phase mixing, while retaining the best paired-view protocol, will exceed 9,315 correct predictions by preserving discriminative within-cell structure until the network learns what to discard.

INTENDED_EDIT: Replace fixed max pooling with two space-to-depth projections that expose every 2×2 sample as channels before learned compression; retain the location-sensitive head and add Reference Design 3’s paired flip supervision and calibrated posterior ensemble. The model has 248,010 learned parameters.

EVIDENCE: Global pooling fell to 9,085 correct and shallow/deep bypass fusion reached only 9,243, showing that spatial information is load-bearing and cannot be recovered merely at the head. The untested shared assumption is that early hard max pooling safely discards spatial phase; learned polyphase mixing challenges it while preserving the 9,315-correct training and inference foundation.

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
            nn.GELU(),
            nn.MaxPool2d(2),
            SpatialRefinement(64),
        )
=======
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
>>>>>>> REPLACE

<<<<<<< SEARCH
        return probabilities.clamp_min(
            torch.finfo(probabilities.dtype).tiny
        ).log()
=======
        return probabilities.clamp_min(
            torch.finfo(probabilities.dtype).tiny
        ).log() / 0.78
>>>>>>> REPLACE

<<<<<<< SEARCH
    del step, total_steps
    return F.cross_entropy(model(images), labels, label_smoothing=0.02)
=======
    del step, total_steps
    paired_images = torch.cat((images, images.flip(-1)), dim=0)
    paired_labels = torch.cat((labels, labels), dim=0)
    return F.cross_entropy(
        model(paired_images),
        paired_labels,
        label_smoothing=0.02,
    )
>>>>>>> REPLACE