MECHANISM: Learned semantic part-token pooling

HYPOTHESIS: Replacing the coordinate-specific flattened head with a deeper residual backbone and content-adaptive global-plus-part tokens will exceed 9,236 correct predictions by learning discriminative regions while making predictions intrinsically less dependent on absolute position.

INTENDED_EDIT: Reallocate the flattened head’s 189,210 parameters into grouped and bottleneck residual feature extraction, then classify one global token and three learned attention-pooled semantic part tokens; preserve the best six-view loss and optimizer to isolate the representational change. The resulting model has 234,357 learned parameters.

EVIDENCE: The 0.75 ensemble-loss design reached 9,236 correct, while nearby weights of 0.70, 0.80, and 0.875 all regressed, indicating that loss-weight refinement has saturated. The best model still spends most of its capacity on an absolute-position flattened head, so adaptive part localization tests a distinct mechanism rather than another ensemble calibration.

<<<<<<< SEARCH
        self.stem = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.GELU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.MaxPool2d(2),
        )
        self.residual = nn.Sequential(
            nn.Conv2d(32, 48, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(48),
            nn.GELU(),
            nn.Conv2d(48, 48, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(48),
        )
        self.shortcut = nn.Sequential(
            nn.Conv2d(32, 48, kernel_size=1, bias=False),
            nn.BatchNorm2d(48),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(48 * 7 * 7, 80),
            nn.GELU(),
            nn.LayerNorm(80),
            nn.Linear(80, 10),
        )
=======
        self.stem = nn.Sequential(
            nn.Conv2d(1, 24, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(24),
            nn.GELU(),
            nn.MaxPool2d(2),
            nn.Conv2d(24, 48, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(48),
            nn.GELU(),
            nn.MaxPool2d(2),
        )
        self.residual_one = nn.Sequential(
            nn.Conv2d(48, 72, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(72),
            nn.GELU(),
            nn.Conv2d(72, 72, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(72),
        )
        self.shortcut_one = nn.Sequential(
            nn.Conv2d(48, 72, kernel_size=1, bias=False),
            nn.BatchNorm2d(72),
        )
        self.residual_two = nn.Sequential(
            nn.Conv2d(72, 96, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(96),
            nn.GELU(),
            nn.Conv2d(
                96,
                96,
                kernel_size=3,
                padding=1,
                groups=4,
                bias=False,
            ),
            nn.BatchNorm2d(96),
        )
        self.shortcut_two = nn.Sequential(
            nn.Conv2d(72, 96, kernel_size=1, bias=False),
            nn.BatchNorm2d(96),
        )
        self.refine = nn.Sequential(
            nn.Conv2d(96, 48, kernel_size=1, bias=False),
            nn.BatchNorm2d(48),
            nn.GELU(),
            nn.Conv2d(
                48,
                48,
                kernel_size=3,
                padding=1,
                groups=48,
                bias=False,
            ),
            nn.BatchNorm2d(48),
            nn.GELU(),
            nn.Conv2d(48, 96, kernel_size=1, bias=False),
            nn.BatchNorm2d(96),
        )
        self.value = nn.Sequential(
            nn.Conv2d(96, 64, kernel_size=1, bias=False),
            nn.BatchNorm2d(64),
            nn.GELU(),
        )
        self.attention = nn.Conv2d(96, 3, kernel_size=1)
        self.classifier = nn.Sequential(
            nn.LayerNorm(4 * 64),
            nn.Linear(4 * 64, 128),
            nn.GELU(),
            nn.LayerNorm(128),
            nn.Linear(128, 10),
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
        features = F.gelu(
            self.residual_one(features) + self.shortcut_one(features)
        )
        features = F.gelu(
            self.residual_two(features) + self.shortcut_two(features)
        )
        features = F.gelu(features + self.refine(features))

        values = self.value(features).flatten(2)
        global_token = values.mean(dim=-1)
        attention = self.attention(features).flatten(2).softmax(dim=-1)
        part_tokens = torch.einsum("bks,bcs->bkc", attention, values)
        tokens = torch.cat((global_token.unsqueeze(1), part_tokens), dim=1)
        return self.classifier(tokens.flatten(1))
>>>>>>> REPLACE