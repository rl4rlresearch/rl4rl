MECHANISM: Residual channel-attention refinement

HYPOTHESIS: Adding lightweight channel attention to Reference Design 3’s residual features will exceed 9,310 correct validation predictions while remaining below 250,000 learned parameters.

INTENDED_EDIT: Adopt the successful two-convolution residual block and add a 580-parameter squeeze-and-excitation gate initialized around identity scaling, yielding 249,973 learned parameters.

EVIDENCE: Reference Design 3 reached the best result of 9,310 correct after deeper spatial refinement improved on the single-convolution design’s 9,287; its 607-parameter headroom motivates testing targeted channel reweighting without sacrificing proven capacity.

<<<<<<< SEARCH
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 48),
            nn.LayerNorm(48),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(48, 10),
        )
=======
        self.refine = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
        )
        self.channel_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(64, 4),
            nn.GELU(),
            nn.Linear(4, 64),
            nn.Sigmoid(),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 35),
            nn.LayerNorm(35),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(35, 10),
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
    def _forward_once(self, images: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(images))
=======
    def _forward_once(self, images: torch.Tensor) -> torch.Tensor:
        features = self.features(images)
        features = F.gelu(features + self.refine(features))
        channel_scale = 2.0 * self.channel_gate(features)
        features = features * channel_scale.view(-1, 64, 1, 1)
        return self.classifier(features)
>>>>>>> REPLACE