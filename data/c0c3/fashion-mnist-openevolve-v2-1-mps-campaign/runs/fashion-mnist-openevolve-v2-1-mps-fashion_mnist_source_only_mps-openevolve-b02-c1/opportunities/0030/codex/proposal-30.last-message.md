MECHANISM: All-convolutional learned downsampling

HYPOTHESIS: Replacing fixed max pooling with overlapping strided convolutions will exceed 9,280 correct predictions by learning which local patterns survive each resolution change while reducing computation enough to avoid the recent timing failures.

INTENDED_EDIT: Move downsampling into the existing refinement convolutions at all three stages, preserving parameter count, output shape, classifier, training procedure, and validation ensemble.

EVIDENCE: The residual refinement kept all fixed max-pooling bottlenecks and fell to 9,266 correct, while added prediction branches repeatedly timed out. This challenges the load-bearing assumption that hard-coded max pooling is the right image representation, using a parameter-neutral mechanism that is also computationally cheaper.

<<<<<<< SEARCH
            nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
=======
            nn.Conv2d(
                32, 32, kernel_size=3, stride=2, padding=1, bias=False
            ),
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
>>>>>>> REPLACE

<<<<<<< SEARCH
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 96, kernel_size=3, padding=1, bias=False),
=======
            nn.Conv2d(
                64, 64, kernel_size=3, stride=2, padding=1, bias=False
            ),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.Conv2d(
                64, 96, kernel_size=3, stride=2, padding=0, bias=False
            ),
>>>>>>> REPLACE

<<<<<<< SEARCH
            nn.BatchNorm2d(96),
            nn.GELU(),
            nn.MaxPool2d(2),
=======
            nn.BatchNorm2d(96),
            nn.GELU(),
>>>>>>> REPLACE