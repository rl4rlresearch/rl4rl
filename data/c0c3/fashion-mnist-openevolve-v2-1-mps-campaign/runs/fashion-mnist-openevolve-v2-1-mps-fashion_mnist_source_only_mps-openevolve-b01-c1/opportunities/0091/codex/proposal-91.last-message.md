MECHANISM: Identity-initialized global channel recalibration

HYPOTHESIS: Adding lightweight image-conditioned channel gating will exceed 9,247 correct predictions by exploiting global garment context without sacrificing the proven spatial representation or materially increasing runtime.

INTENDED_EDIT: Add a zero-initialized squeeze-and-excitation-style gate after the residual extractor, allowing each image to modulate its 48 feature channels within a bounded range while keeping the model initially identical to the baseline.

EVIDENCE: Additional spatial convolution regressed to 9,228 correct and a larger flattened head regressed to 9,210, while global covariance pooling timed out; this motivates a computationally cheap global-context mechanism that adds neither spatial refinement nor substantial decoder capacity.

<<<<<<< SEARCH
        self.shortcut = nn.Sequential(
            nn.Conv2d(32, 48, kernel_size=1, bias=False),
            nn.BatchNorm2d(48),
        )
        self.classifier = nn.Sequential(
=======
        self.shortcut = nn.Sequential(
            nn.Conv2d(32, 48, kernel_size=1, bias=False),
            nn.BatchNorm2d(48),
        )
        self.channel_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(48, 24, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(24, 48, kernel_size=1),
        )
        nn.init.zeros_(self.channel_gate[-1].weight)
        nn.init.zeros_(self.channel_gate[-1].bias)
        self.classifier = nn.Sequential(
>>>>>>> REPLACE

<<<<<<< SEARCH
        features = self.stem(images)
        features = F.gelu(self.residual(features) + self.shortcut(features))
        return self.classifier(features)
=======
        features = self.stem(images)
        features = F.gelu(self.residual(features) + self.shortcut(features))
        gate = torch.tanh(self.channel_gate(features))
        features = features * (1.0 + 0.5 * gate)
        return self.classifier(features)
>>>>>>> REPLACE