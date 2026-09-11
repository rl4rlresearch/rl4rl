MECHANISM: Learned convolutional downsampling with capacity rebalancing

HYPOTHESIS: Reallocating parameters from the flattening dense head into a 96-channel learned spatial downsampling stage will exceed 9,260 correct predictions by extracting stronger local features within the fixed exposure budget.

INTENDED_EDIT: Add a stride-2 convolutional stage after the existing feature extractor and resize the dense head, increasing learned parameters from 239,634 to 249,808 while preserving training and calibration settings.

EVIDENCE: Dense-head dropout reduced correctness to 9,237 and a projected residual rewiring reached only 9,246; since augmentation and label-smoothing changes also failed, redistributing the parameter-dominant dense capacity into additional spatial feature learning is the most distinct architecture test.

<<<<<<< SEARCH
            nn.GELU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(48 * 7 * 7, 88),
            nn.GELU(),
            nn.Linear(88, 10),
        )
=======
            nn.GELU(),
            nn.MaxPool2d(2),
            nn.Conv2d(48, 96, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(96),
            nn.GELU(),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(96 * 4 * 4, 114),
            nn.GELU(),
            nn.Linear(114, 10),
        )
>>>>>>> REPLACE