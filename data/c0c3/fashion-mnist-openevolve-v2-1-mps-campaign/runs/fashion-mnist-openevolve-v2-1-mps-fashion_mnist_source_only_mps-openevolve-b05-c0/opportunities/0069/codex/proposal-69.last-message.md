MECHANISM: Channel-wise parametric rectification with inference sharpening

HYPOTHESIS: Replacing spatial GELUs with channel-wise PReLUs will recover the accuracy lost by fixed ReLUs while retaining their sub-75-second runtime, and 1.2× evaluation sharpening will improve cross-entropy without changing predicted classes.

INTENDED_EDIT: Use learned per-channel negative slopes in all three convolutional blocks, retain the classifier GELU, and raise the evaluation-only logit multiplier from 1.1 to 1.2.

EVIDENCE: Fixed spatial ReLUs finished in 68.65 seconds but lost 23 correct predictions versus exact GELUs; PReLU preserves that inexpensive rectified computation while learning the negative response that ReLU removes. Prior 1.1× sharpening improved cross-entropy without changing argmax predictions, while 1.2× attempts timed out without contradictory calibration evidence.

<<<<<<< SEARCH
            nn.Conv2d(1, 24, kernel_size=3, padding=1),
            nn.BatchNorm2d(24),
            nn.GELU(),
            nn.MaxPool2d(2),
            nn.Conv2d(24, 48, kernel_size=3, padding=1),
            nn.BatchNorm2d(48),
            nn.GELU(),
            nn.Conv2d(48, 48, kernel_size=3, padding=1),
            nn.BatchNorm2d(48),
            nn.GELU(),
            nn.MaxPool2d(2),
=======
            nn.Conv2d(1, 24, kernel_size=3, padding=1),
            nn.BatchNorm2d(24),
            nn.PReLU(num_parameters=24),
            nn.MaxPool2d(2),
            nn.Conv2d(24, 48, kernel_size=3, padding=1),
            nn.BatchNorm2d(48),
            nn.PReLU(num_parameters=48),
            nn.Conv2d(48, 48, kernel_size=3, padding=1),
            nn.BatchNorm2d(48),
            nn.PReLU(num_parameters=48),
            nn.MaxPool2d(2),
>>>>>>> REPLACE

<<<<<<< SEARCH
        if not self.training:
            logits = logits * 1.1
=======
        if not self.training:
            logits = logits * 1.2
>>>>>>> REPLACE