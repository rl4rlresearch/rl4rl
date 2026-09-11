MECHANISM: Learned convolutional downsampling with capacity rebalancing

HYPOTHESIS: A 96-channel spatial downsampling stage and resized dense head will exceed 9,260 correct predictions by allocating more capacity to local feature extraction while remaining below 250,000 parameters.

INTENDED_EDIT: Add a stride-2 48→96 convolution after the existing feature extractor and change the classifier to a 1,536→114→10 head, yielding 249,808 learned parameters.

EVIDENCE: The previous verification of this architecture timed out without validation evidence; because label-smoothing, augmentation, dropout, and residual-block changes all reduced correctness, retrying this distinct capacity redistribution remains the most informative next test.

<<<<<<< SEARCH
            nn.Conv2d(48, 48, kernel_size=3, padding=1),
            nn.BatchNorm2d(48),
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
            nn.Conv2d(48, 48, kernel_size=3, padding=1),
            nn.BatchNorm2d(48),
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