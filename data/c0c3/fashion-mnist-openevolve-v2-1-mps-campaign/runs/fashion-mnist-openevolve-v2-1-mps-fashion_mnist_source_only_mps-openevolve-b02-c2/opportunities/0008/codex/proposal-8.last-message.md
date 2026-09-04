MECHANISM: Identity-initialized channel-spatial attention

HYPOTHESIS: Restoring the 9,245-correct second bottleneck and adding identity-initialized channel-spatial attention will exceed 9,245 correct predictions while remaining below 250,000 learned parameters.

INTENDED_EDIT: Add Reference Design 2’s second 24-channel residual bottleneck, followed by zero-initialized ECA-style channel attention and spatial attention, for approximately 249,855 parameters.

EVIDENCE: The second low-resolution bottleneck improved 9,236 to 9,245 correct, whereas EMA reduced accuracy; dynamic feature selection uses the remaining 246-parameter headroom while preserving the best verified optimizer, augmentation, and evaluation regimen.

<<<<<<< SEARCH
        self.deep_conv3 = nn.Conv2d(32, 64, kernel_size=1, bias=False)
        self.deep_norm3 = nn.BatchNorm2d(64)

        self.classifier = nn.Sequential(
=======
        self.deep_conv3 = nn.Conv2d(32, 64, kernel_size=1, bias=False)
        self.deep_norm3 = nn.BatchNorm2d(64)

        self.post_deep_conv1 = nn.Conv2d(64, 24, kernel_size=1, bias=False)
        self.post_deep_norm1 = nn.BatchNorm2d(24)
        self.post_deep_conv2 = nn.Conv2d(
            24, 24, kernel_size=3, padding=1, bias=False
        )
        self.post_deep_norm2 = nn.BatchNorm2d(24)
        self.post_deep_conv3 = nn.Conv2d(24, 64, kernel_size=1, bias=False)
        self.post_deep_norm3 = nn.BatchNorm2d(64)

        self.channel_attention = nn.Conv1d(
            1, 1, kernel_size=3, padding=1, bias=False
        )
        self.spatial_attention = nn.Conv2d(
            2, 1, kernel_size=7, padding=3, bias=False
        )
        nn.init.zeros_(self.channel_attention.weight)
        nn.init.zeros_(self.spatial_attention.weight)

        self.classifier = nn.Sequential(
>>>>>>> REPLACE

<<<<<<< SEARCH
        features = self.deep_norm3(self.deep_conv3(features))
        features = F.gelu(features + residual)
        return self.classifier(features)
=======
        features = self.deep_norm3(self.deep_conv3(features))
        features = F.gelu(features + residual)

        residual = features
        features = F.gelu(
            self.post_deep_norm1(self.post_deep_conv1(features))
        )
        features = F.gelu(
            self.post_deep_norm2(self.post_deep_conv2(features))
        )
        features = self.post_deep_norm3(self.post_deep_conv3(features))
        features = F.gelu(features + residual)

        channel_summary = F.adaptive_avg_pool2d(features, 1)
        channel_summary = channel_summary.flatten(2).transpose(1, 2)
        channel_gate = self.channel_attention(channel_summary)
        channel_gate = channel_gate.transpose(1, 2).unsqueeze(-1)
        features = features * (2.0 * torch.sigmoid(channel_gate))

        spatial_summary = torch.cat(
            (
                features.mean(dim=1, keepdim=True),
                features.amax(dim=1, keepdim=True),
            ),
            dim=1,
        )
        spatial_gate = self.spatial_attention(spatial_summary)
        features = features * (2.0 * torch.sigmoid(spatial_gate))
        return self.classifier(features)
>>>>>>> REPLACE