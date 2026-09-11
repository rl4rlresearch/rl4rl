MECHANISM: Identity-initialized squeeze-and-excitation channel gating

HYPOTHESIS: Context-dependent channel reweighting will exceed 9,257 correct predictions by exploiting global garment context while preserving the proven coordinate-specific classifier.

INTENDED_EDIT: Add a 960-parameter global-average channel gate after the residual stage, initialized to an exact identity so optimization begins from the current model.

EVIDENCE: A global-average classification branch reached 9,253 correct, suggesting pooled context contains useful but insufficient standalone evidence; using it to modulate spatial features retains the stronger flattened head.

<<<<<<< SEARCH
        self.shortcut = nn.Sequential(
            nn.Conv2d(40, 56, kernel_size=1, bias=False),
            nn.BatchNorm2d(56),
        )
        self.classifier = nn.Sequential(
=======
        self.shortcut = nn.Sequential(
            nn.Conv2d(40, 56, kernel_size=1, bias=False),
            nn.BatchNorm2d(56),
        )
        self.channel_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(56, 8, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(8, 56, kernel_size=1),
        )
        nn.init.zeros_(self.channel_gate[-1].weight)
        nn.init.zeros_(self.channel_gate[-1].bias)
        self.classifier = nn.Sequential(
>>>>>>> REPLACE

<<<<<<< SEARCH
        features = F.gelu(self.residual(features) + self.shortcut(features))
        return self.classifier(features)
=======
        features = F.gelu(self.residual(features) + self.shortcut(features))
        gate = 2.0 * torch.sigmoid(self.channel_gate(features))
        return self.classifier(features * gate)
>>>>>>> REPLACE