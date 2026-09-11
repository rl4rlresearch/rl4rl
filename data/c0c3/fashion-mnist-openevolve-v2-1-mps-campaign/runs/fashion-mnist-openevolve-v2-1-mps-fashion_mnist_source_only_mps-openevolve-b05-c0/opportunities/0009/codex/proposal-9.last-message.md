MECHANISM: Per-layer convolutional activation normalization

HYPOTHESIS: Adding BatchNorm after each convolution will exceed 9,170 correct predictions by improving optimization during the fixed 782-step budget while preserving the successful architecture, inputs, loss, schedule, and EMA.

INTENDED_EDIT: Insert BatchNorm2d between each convolution and GELU, increasing learned parameters from 239,394 to 239,634.

EVIDENCE: The unaugmented three-convolution model reached 9,170 correct, whereas the prior BatchNorm residual model was confounded by a different architecture plus harmful crops and flips; isolating normalization is the most informative next test.

<<<<<<< SEARCH
            nn.Conv2d(1, 24, kernel_size=3, padding=1),
            nn.GELU(),
            nn.MaxPool2d(2),
            nn.Conv2d(24, 48, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(48, 48, kernel_size=3, padding=1),
            nn.GELU(),
            nn.MaxPool2d(2),
=======
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
>>>>>>> REPLACE