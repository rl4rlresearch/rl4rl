MECHANISM: Fixed-slope leaky rectification with inference sharpening

HYPOTHESIS: Spatial LeakyReLUs with a 0.1 negative slope will finish reliably and recover the accuracy lost by ReLU, retaining negative activations without PReLU’s channel-wise overhead; 1.2× evaluation logits will reduce cross-entropy without changing predicted classes.

INTENDED_EDIT: Replace all three convolutional GELUs with in-place LeakyReLUs while retaining the classifier GELU, and increase inference-only logit scaling from 1.1 to 1.2.

EVIDENCE: Spatial ReLUs finished in 68.65 seconds but scored 9,235 correct, while exact GELUs scored 9,258 but took 80.92 seconds. The channel-wise PReLU attempt timed out, motivating the cheaper fixed-slope form; prior 1.1× sharpening improved cross-entropy without changing argmax predictions.

<<<<<<< SEARCH
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
            nn.BatchNorm2d(24),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(24, 48, kernel_size=3, padding=1),
            nn.BatchNorm2d(48),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(48, 48, kernel_size=3, padding=1),
            nn.BatchNorm2d(48),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.MaxPool2d(2),
>>>>>>> REPLACE

<<<<<<< SEARCH
        if not self.training:
            logits = logits * 1.1
=======
        if not self.training:
            logits = logits * 1.2
>>>>>>> REPLACE