MECHANISM: Multi-scale convolutional evidence readout

HYPOTHESIS: Replacing the shared 48-dimensional global bottleneck with multi-scale local context features and a direct class-specific spatial readout will exceed 9,316 correct predictions by preserving discriminative spatial evidence until the final logits.

INTENDED_EDIT: Retain the successful invariant–disagreement fusion, but replace its flattened MLP classifier with lightweight depthwise context branches at two receptive-field scales, channel mixing, and direct per-class spatial templates.

EVIDENCE: Joint view fusion reached 9,316 correct, but widening the general classifier fell to 9,254 and several increasingly elaborate fusion mechanisms failed or regressed. This challenges the shared assumption that fused features should first collapse into one small global hidden vector, testing a different prediction mechanism without materially increasing runtime.

<<<<<<< SEARCH
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 48),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(48, 10),
        )
=======
        self.context_near = nn.Sequential(
            nn.Conv2d(
                64, 64, kernel_size=3, padding=1,
                groups=64, bias=False,
            ),
            nn.BatchNorm2d(64),
            nn.GELU(),
        )
        self.context_wide = nn.Sequential(
            nn.Conv2d(
                64, 64, kernel_size=3, padding=2, dilation=2,
                groups=64, bias=False,
            ),
            nn.BatchNorm2d(64),
            nn.GELU(),
        )
        self.context_mix = nn.Sequential(
            nn.Conv2d(192, 96, kernel_size=1, bias=False),
            nn.BatchNorm2d(96),
            nn.GELU(),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.15),
            nn.Linear(96 * 7 * 7, 10),
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
        return self.classifier(fused)
=======
        context = torch.cat(
            (
                fused,
                self.context_near(fused),
                self.context_wide(fused),
            ),
            dim=1,
        )
        return self.classifier(self.context_mix(context))
>>>>>>> REPLACE