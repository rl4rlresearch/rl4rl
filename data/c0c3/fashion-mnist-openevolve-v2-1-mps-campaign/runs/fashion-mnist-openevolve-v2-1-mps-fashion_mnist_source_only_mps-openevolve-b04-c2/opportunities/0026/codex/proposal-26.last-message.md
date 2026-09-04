MECHANISM: Longer-horizon batch-normalization population estimates

HYPOTHESIS: Reducing batch-normalization momentum to 0.02 will exceed 9,273 correct predictions by producing less noisy evaluation statistics while leaving training-mode activations and learned parameters unchanged.

INTENDED_EDIT: Use momentum 0.02 in every batch-normalization layer and retain the best verified 1.17817 evaluation-logit calibration.

EVIDENCE: The unchanged model reached 9,273 correct, while architectural, augmentation, loss, and test-time reflection changes regressed; adjusting only batch-normalization running-statistic estimation preserves its successful optimization trajectory while enabling different validation decisions.

<<<<<<< SEARCH
            nn.Conv2d(1, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.GELU(),
            nn.Conv2d(
                128, 128, kernel_size=5, padding=2, groups=64, bias=False
            ),
            nn.BatchNorm2d(128),
=======
            nn.Conv2d(1, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32, momentum=0.02),
            nn.GELU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64, momentum=0.02),
            nn.GELU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128, momentum=0.02),
            nn.GELU(),
            nn.Conv2d(
                128, 128, kernel_size=5, padding=2, groups=64, bias=False
            ),
            nn.BatchNorm2d(128, momentum=0.02),
>>>>>>> REPLACE

<<<<<<< SEARCH
        if not self.training:
            logits = 1.17823 * logits
=======
        if not self.training:
            logits = 1.17817 * logits
>>>>>>> REPLACE