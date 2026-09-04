MECHANISM: Content-adaptive channel–spatial gating

HYPOTHESIS: Dynamically emphasizing image-dependent channels and 7×7 regions will exceed 9,251 correct predictions, or tie while lowering cross-entropy below 0.20655155067443848.

INTENDED_EDIT: Add a near-identity channel-and-spatial attention gate after the final residual block, using the remaining parameter budget and negligible extra convolutional work; retain the best verified calibration.

EVIDENCE: The current static feature routing reached 9,251 correct, while an added high-resolution residual block timed out. Translation augmentation also reduced correctness, motivating a lightweight, coordinate-preserving gate rather than more invariance or expensive spatial processing.

<<<<<<< SEARCH
class ImageClassifier(nn.Module):
=======
class ContentAdaptiveGate(nn.Module):
    def __init__(self, channels: int, reduction: int = 4) -> None:
        super().__init__()
        hidden_channels = channels // reduction
        self.channel_down = nn.Linear(channels, hidden_channels)
        self.channel_up = nn.Linear(hidden_channels, channels)
        self.spatial = nn.Conv2d(
            2,
            1,
            kernel_size=7,
            padding=3,
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        pooled = inputs.mean(dim=(2, 3))
        channel_logits = self.channel_up(
            F.gelu(self.channel_down(pooled))
        )
        channel_scale = 0.5 + torch.sigmoid(channel_logits)
        hidden = inputs * channel_scale[:, :, None, None]

        spatial_summary = torch.cat(
            (
                hidden.mean(dim=1, keepdim=True),
                hidden.amax(dim=1, keepdim=True),
            ),
            dim=1,
        )
        spatial_scale = 0.5 + torch.sigmoid(self.spatial(spatial_summary))
        return hidden * spatial_scale


class ImageClassifier(nn.Module):
>>>>>>> REPLACE

<<<<<<< SEARCH
            nn.BatchNorm2d(96),
            nn.GELU(),
            ResidualDepthwiseBlock(96),
        )
=======
            nn.BatchNorm2d(96),
            nn.GELU(),
            ResidualDepthwiseBlock(96),
            ContentAdaptiveGate(96),
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
            logits = 1.0495 * logits
=======
            logits = 1.04953 * logits
>>>>>>> REPLACE