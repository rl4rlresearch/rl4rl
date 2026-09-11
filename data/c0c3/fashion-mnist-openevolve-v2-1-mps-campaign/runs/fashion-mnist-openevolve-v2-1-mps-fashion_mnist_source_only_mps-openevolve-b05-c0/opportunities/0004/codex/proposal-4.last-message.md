MECHANISM: Conservative width-and-depth scaling without augmentation

HYPOTHESIS: A 239,394-parameter three-convolution extension of the verified baseline, trained with batch size 128 for 782 optimizer steps, will exceed 8,928 correct validation predictions.

INTENDED_EDIT: Widen the baseline to 24/48 channels, add one 48-channel convolution, use an 88-unit classifier, and halve the batch size while retaining the verified loss and optimizer schedule.

EVIDENCE: The 241,130-parameter residual model with crops, flips, and BatchNorm fell to 8,705 correct; this motivates preserving the baseline’s simple unaugmented training path while testing additional capacity and updates without those confounding changes.

<<<<<<< SEARCH
BATCH_SIZE = 256
GRAD_CLIP_NORM = 1.0
=======
BATCH_SIZE = 128
GRAD_CLIP_NORM = 1.0
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.GELU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.GELU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 7 * 7, 64),
            nn.GELU(),
            nn.Linear(64, 10),
        )
=======
        self.features = nn.Sequential(
            nn.Conv2d(1, 24, kernel_size=3, padding=1),
            nn.GELU(),
            nn.MaxPool2d(2),
            nn.Conv2d(24, 48, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(48, 48, kernel_size=3, padding=1),
            nn.GELU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(48 * 7 * 7, 88),
            nn.GELU(),
            nn.Linear(88, 10),
        )
>>>>>>> REPLACE