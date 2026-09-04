MECHANISM: Residual high-resolution class-evidence branch

HYPOTHESIS: Summing class-specific 7×7 spatial evidence with the existing coarse 3×3 nonlinear prediction path will exceed 9,210 correct predictions while staying below the parameter ceiling.

INTENDED_EDIT: Expose the unpooled third-stage feature map, project it to eight channels, and add its direct class logits to a slightly narrowed 155-unit coarse classifier; total learned parameters are 249,936.

EVIDENCE: Full-grid classification reached 9,210 correct while global-context and axial-profile alternatives reached only 9,128 and 9,167, showing that 2D layout is load-bearing. The failed GEGLU head still operated solely after the 7×7-to-3×3 max-pooling bottleneck; this patch instead challenges the assumption that all useful spatial evidence survives that bottleneck.

<<<<<<< SEARCH
        self.features = nn.Sequential(
            nn.Conv2d(4, 24, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(24),
            nn.GELU(),
            nn.MaxPool2d(2),
            nn.Conv2d(24, 48, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(48),
            nn.GELU(),
            nn.MaxPool2d(2),
            nn.Conv2d(48, 72, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(72),
            nn.GELU(),
            nn.MaxPool2d(2),
            nn.Conv2d(72, 96, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(96),
            nn.GELU(),
        )
        self.channel_gate = nn.Sequential(
            nn.Linear(96, 24),
            nn.GELU(),
            nn.Linear(24, 96),
        )
        nn.init.zeros_(self.channel_gate[-1].weight)
        nn.init.zeros_(self.channel_gate[-1].bias)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.15),
            nn.Linear(96 * 3 * 3, 160),
            nn.GELU(),
            nn.Dropout(0.10),
            nn.Linear(160, 10),
        )
=======
        self.stage1 = nn.Sequential(
            nn.Conv2d(4, 24, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(24),
            nn.GELU(),
            nn.MaxPool2d(2),
        )
        self.stage2 = nn.Sequential(
            nn.Conv2d(24, 48, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(48),
            nn.GELU(),
            nn.MaxPool2d(2),
        )
        self.stage3 = nn.Sequential(
            nn.Conv2d(48, 72, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(72),
            nn.GELU(),
        )
        self.stage3_pool = nn.MaxPool2d(2)
        self.stage4 = nn.Sequential(
            nn.Conv2d(72, 96, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(96),
            nn.GELU(),
        )
        self.channel_gate = nn.Sequential(
            nn.Linear(96, 24),
            nn.GELU(),
            nn.Linear(24, 96),
        )
        nn.init.zeros_(self.channel_gate[-1].weight)
        nn.init.zeros_(self.channel_gate[-1].bias)
        self.coarse_classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.15),
            nn.Linear(96 * 3 * 3, 155),
            nn.GELU(),
            nn.Dropout(0.10),
            nn.Linear(155, 10),
        )
        self.fine_projection = nn.Sequential(
            nn.Conv2d(72, 8, kernel_size=1, bias=False),
            nn.BatchNorm2d(8),
            nn.GELU(),
        )
        self.fine_classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.10),
            nn.Linear(8 * 7 * 7, 10),
        )
        with torch.no_grad():
            self.fine_classifier[-1].weight.mul_(0.25)
            self.fine_classifier[-1].bias.zero_()
>>>>>>> REPLACE

<<<<<<< SEARCH
        represented = torch.cat((images, details), dim=1)
        features = self.features(represented)
        average_context = features.mean(dim=(2, 3))
        maximum_context = features.amax(dim=(2, 3))
        channel_signal = 0.5 * (
            self.channel_gate(average_context)
            + self.channel_gate(maximum_context)
        )
        channel_scale = 1.0 + 0.5 * torch.tanh(channel_signal)
        features = features * channel_scale[:, :, None, None]
        return self.classifier(features)
=======
        represented = torch.cat((images, details), dim=1)
        features = self.stage1(represented)
        features = self.stage2(features)
        fine_features = self.stage3(features)
        coarse_features = self.stage4(self.stage3_pool(fine_features))
        average_context = coarse_features.mean(dim=(2, 3))
        maximum_context = coarse_features.amax(dim=(2, 3))
        channel_signal = 0.5 * (
            self.channel_gate(average_context)
            + self.channel_gate(maximum_context)
        )
        channel_scale = 1.0 + 0.5 * torch.tanh(channel_signal)
        coarse_features = coarse_features * channel_scale[:, :, None, None]
        coarse_logits = self.coarse_classifier(coarse_features)
        fine_logits = self.fine_classifier(self.fine_projection(fine_features))
        return coarse_logits + fine_logits
>>>>>>> REPLACE