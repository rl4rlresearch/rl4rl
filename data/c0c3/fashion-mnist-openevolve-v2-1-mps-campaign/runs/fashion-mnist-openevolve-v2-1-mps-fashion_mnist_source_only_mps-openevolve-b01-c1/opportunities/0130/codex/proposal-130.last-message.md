MECHANISM: Context-gated residual refinement

HYPOTHESIS: Reallocating classifier-head parameters into a second residual block with input-conditioned channel gating will exceed 9,262 correct predictions by learning richer spatial features while retaining the coordinate-specific layout information lost by prior pooling designs.

INTENDED_EDIT: Reduce the flattened bottleneck from 64 to 43 units and use the recovered budget for a 56-channel residual refinement block whose channels are gated by global image context, yielding approximately 233,847 learned parameters.

EVIDENCE: Widening the coordinate-specific head to 249,976 parameters regressed to 9,221 correct, showing that additional head capacity is ineffective. Global pooling reached 9,253 but discarded useful spatial layout; using global statistics only to modulate a preserved 7×7 feature map tests context-aware representation without that information loss.

<<<<<<< SEARCH
        self.shortcut = nn.Sequential(
            nn.Conv2d(40, 56, kernel_size=1, bias=False),
            nn.BatchNorm2d(56),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(56 * 7 * 7, 64),
            nn.GELU(),
            nn.LayerNorm(64),
            nn.Linear(64, 10),
        )
=======
        self.shortcut = nn.Sequential(
            nn.Conv2d(40, 56, kernel_size=1, bias=False),
            nn.BatchNorm2d(56),
        )
        self.refine = nn.Sequential(
            nn.Conv2d(56, 56, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(56),
            nn.GELU(),
            nn.Conv2d(56, 56, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(56),
        )
        self.channel_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(56, 14),
            nn.GELU(),
            nn.Linear(14, 56),
            nn.Sigmoid(),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(56 * 7 * 7, 43),
            nn.GELU(),
            nn.LayerNorm(43),
            nn.Linear(43, 10),
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
    def _forward_once(self, images: torch.Tensor) -> torch.Tensor:
        features = self.stem(images)
        features = F.gelu(self.residual(features) + self.shortcut(features))
        return self.classifier(features)
=======
    def _forward_once(self, images: torch.Tensor) -> torch.Tensor:
        features = self.stem(images)
        features = F.gelu(self.residual(features) + self.shortcut(features))
        refined = self.refine(features)
        gate = self.channel_gate(refined).unsqueeze(-1).unsqueeze(-1)
        features = F.gelu(features + gate * refined)
        return self.classifier(features)
>>>>>>> REPLACE