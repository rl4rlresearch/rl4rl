MECHANISM: Global-context gated residual features with multi-head attentive shape pooling

HYPOTHESIS: Replacing the coordinate-specific flattened head and 50-view compensation with deeper low-resolution, multi-scale features and learned global pooling will exceed 9,063 correct predictions.

INTENDED_EDIT: Replace the model with a 242,098-parameter dilated bottleneck CNN using image-conditioned channel gates, four spatial-attention heads, coarse spatial pooling, and an 18-view central translation/flip ensemble.

EVIDENCE: The 18-view ensemble already reached 9,055 correct with lower cross-entropy than the 50-view version, while the current model spends most of its capacity on an absolute-position linear head; this challenges the old assumption that brute-force translations are preferable to learning global shape aggregation.

<<<<<<< SEARCH
class ImageClassifier(nn.Module):
    def __init__(self) -> None:
        super().__init__()
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

    def _forward_once(self, images: torch.Tensor) -> torch.Tensor:
        features = self.stem(images)
        features = F.gelu(self.residual(features) + self.shortcut(features))
        return self.classifier(features)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        if self.training:
            return self._forward_once(images)
        padded = F.pad(images, (2, 2, 2, 2), mode="reflect")
        logits_sum = None
        central_logits_sum = None
        for offset_y in range(5):
            for offset_x in range(5):
                view = padded[
                    :,
                    :,
                    offset_y : offset_y + 28,
                    offset_x : offset_x + 28,
                ]
                views = torch.cat((view, view.flip(-1)), dim=0)
                logits = self._forward_once(views)
                original_logits, flipped_logits = logits.chunk(2, dim=0)
                view_logits = original_logits + flipped_logits
                if logits_sum is None:
                    logits_sum = view_logits
                else:
                    logits_sum = logits_sum + view_logits
                if 1 <= offset_y <= 3 and 1 <= offset_x <= 3:
                    if central_logits_sum is None:
                        central_logits_sum = view_logits
                    else:
                        central_logits_sum = central_logits_sum + view_logits
        full_ensemble = logits_sum / 50.0
        central_ensemble = central_logits_sum / 18.0
        return 0.9 * full_ensemble + 0.1 * central_ensemble
=======
class _GlobalChannelGate(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        hidden_channels = max(channels // 8, 8)
        self.reduce = nn.Linear(channels, hidden_channels)
        self.expand = nn.Linear(hidden_channels, channels)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        context = F.adaptive_avg_pool2d(features, 1).flatten(1)
        gates = self.expand(F.gelu(self.reduce(context)))
        gates = 2.0 * torch.sigmoid(gates).unsqueeze(-1).unsqueeze(-1)
        return features * gates


class _ContextResidualBlock(nn.Module):
    def __init__(
        self,
        input_channels: int,
        output_channels: int,
        bottleneck_channels: int,
        dilation: int,
    ) -> None:
        super().__init__()
        self.reduce = nn.Sequential(
            nn.Conv2d(
                input_channels,
                bottleneck_channels,
                kernel_size=1,
                bias=False,
            ),
            nn.BatchNorm2d(bottleneck_channels),
            nn.GELU(),
        )
        self.spatial = nn.Sequential(
            nn.Conv2d(
                bottleneck_channels,
                bottleneck_channels,
                kernel_size=3,
                padding=dilation,
                dilation=dilation,
                bias=False,
            ),
            nn.BatchNorm2d(bottleneck_channels),
            nn.GELU(),
        )
        self.expand = nn.Sequential(
            nn.Conv2d(
                bottleneck_channels,
                output_channels,
                kernel_size=1,
                bias=False,
            ),
            nn.BatchNorm2d(output_channels),
        )
        self.gate = _GlobalChannelGate(output_channels)
        if input_channels == output_channels:
            self.shortcut = nn.Identity()
        else:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    input_channels,
                    output_channels,
                    kernel_size=1,
                    bias=False,
                ),
                nn.BatchNorm2d(output_channels),
            )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        residual = self.expand(self.spatial(self.reduce(features)))
        residual = self.gate(residual)
        return F.gelu(residual + self.shortcut(features))


class ImageClassifier(nn.Module):
    def __init__(self) -> None:
        super().__init__()
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
        self.blocks = nn.ModuleList(
            (
                _ContextResidualBlock(32, 72, 56, dilation=1),
                _ContextResidualBlock(72, 72, 56, dilation=2),
                _ContextResidualBlock(72, 72, 56, dilation=1),
                _ContextResidualBlock(72, 72, 56, dilation=2),
            )
        )
        self.spatial_attention = nn.Conv2d(
            72,
            4,
            kernel_size=1,
        )
        self.classifier = nn.Sequential(
            nn.Linear(72 * 9, 128),
            nn.GELU(),
            nn.LayerNorm(128),
            nn.Linear(128, 10),
        )

    def _forward_once(self, images: torch.Tensor) -> torch.Tensor:
        features = self.stem(images)
        for block in self.blocks:
            features = block(features)

        coarse_shape = F.adaptive_avg_pool2d(features, 2).flatten(1)
        peak_features = F.adaptive_max_pool2d(features, 1).flatten(1)

        attention = self.spatial_attention(features).flatten(2)
        attention = F.softmax(attention, dim=-1)
        feature_tokens = features.flatten(2).unsqueeze(1)
        attended_features = (
            feature_tokens * attention.unsqueeze(2)
        ).sum(dim=-1).flatten(1)

        descriptor = torch.cat(
            (coarse_shape, peak_features, attended_features),
            dim=1,
        )
        return self.classifier(descriptor)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        if self.training:
            return self._forward_once(images)

        padded = F.pad(images, (1, 1, 1, 1), mode="reflect")
        logits_sum = None
        for offset_y in range(3):
            for offset_x in range(3):
                view = padded[
                    :,
                    :,
                    offset_y : offset_y + 28,
                    offset_x : offset_x + 28,
                ]
                views = torch.cat((view, view.flip(-1)), dim=0)
                logits = self._forward_once(views)
                original_logits, flipped_logits = logits.chunk(2, dim=0)
                view_logits = original_logits + flipped_logits
                if logits_sum is None:
                    logits_sum = view_logits
                else:
                    logits_sum = logits_sum + view_logits
        return logits_sum / 18.0
>>>>>>> REPLACE