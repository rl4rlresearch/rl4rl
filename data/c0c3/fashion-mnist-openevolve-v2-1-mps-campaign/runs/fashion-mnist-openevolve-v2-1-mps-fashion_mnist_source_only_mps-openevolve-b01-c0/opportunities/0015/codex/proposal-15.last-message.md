MECHANISM: Channel-projected wider dense bottleneck

HYPOTHESIS: A learned 64→48 pointwise projection followed by a 76-unit dense bottleneck will exceed 9,290 correct predictions by providing greater nonlinear head width within the parameter ceiling while retaining the successful convolutional extractor.

INTENDED_EDIT: Insert a normalized pointwise channel projection before flattening and widen the classifier bottleneck from 58 to 76 units, increasing the model to 248,102 learned parameters while leaving training, augmentation, optimization, and ensembling unchanged.

EVIDENCE: Expanding the dense bottleneck from 48 to 58 improved validation correct from 9,265 to 9,290, whereas adding a fifth convolution while narrowing the bottleneck to 36 regressed to 9,157; this motivates a parameter-efficient way to extend dense-head width without replacing the verified extractor.

<<<<<<< SEARCH
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 58),
            nn.BatchNorm1d(58),
            nn.SiLU(),
            nn.Dropout(p=0.1),
            nn.Linear(58, 10),
        )
=======
        self.projection = nn.Sequential(
            nn.Conv2d(64, 48, kernel_size=1, bias=False),
            nn.BatchNorm2d(48),
            nn.SiLU(),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(48 * 7 * 7, 76),
            nn.BatchNorm1d(76),
            nn.SiLU(),
            nn.Dropout(p=0.1),
            nn.Linear(76, 10),
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
    def _forward_once(self, images: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(images))
=======
    def _forward_once(self, images: torch.Tensor) -> torch.Tensor:
        features = self.projection(self.features(images))
        return self.classifier(features)
>>>>>>> REPLACE