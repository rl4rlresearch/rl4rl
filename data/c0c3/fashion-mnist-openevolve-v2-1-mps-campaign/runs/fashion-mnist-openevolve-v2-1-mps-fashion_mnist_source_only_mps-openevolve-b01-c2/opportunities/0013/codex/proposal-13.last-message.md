MECHANISM: Stacked single-convolution residual refinement

HYPOTHESIS: Giving each proven refinement convolution its own identity shortcut at batch size 64 will exceed 9,312 correct predictions by improving gradient flow during the fixed two-pass exposure.

INTENDED_EDIT: Remove harmful channel attention, restore batch size 64, and reorganize the parameter-identical two-convolution refinement into two sequential residual units.

EVIDENCE: The 249,393-parameter batch-64 design achieved the best result at 9,312 correct, while channel attention reduced accuracy to 9,296; earlier evidence also showed that residual spatial refinement improved from 9,287 to 9,310 when deepened.

<<<<<<< SEARCH
BATCH_SIZE = 128
=======
BATCH_SIZE = 64
>>>>>>> REPLACE

<<<<<<< SEARCH
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
=======
        self.refine1 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
        )
        self.refine2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
    def _forward_once(self, images: torch.Tensor) -> torch.Tensor:
        features = self.features(images)
        features = F.gelu(features + self.refine(features))
        channel_scale = 2.0 * self.channel_gate(features)
        features = features * channel_scale.view(-1, 64, 1, 1)
        return self.classifier(features)
=======
    def _forward_once(self, images: torch.Tensor) -> torch.Tensor:
        features = self.features(images)
        features = F.gelu(features + self.refine1(features))
        features = F.gelu(features + self.refine2(features))
        return self.classifier(features)
>>>>>>> REPLACE