MECHANISM: Content-addressed multi-part attention pooling

HYPOTHESIS: Replacing the fixed-position 30-feature bottleneck with seven learned spatial part summaries plus a global summary will exceed 9,240 correct predictions by preserving substantially more channel information while making prediction less dependent on exact 7×7 coordinates.

INTENDED_EDIT: Replace the 94,420-parameter flattened classifier with a 235,793-total-parameter attention-pooling model that dynamically locates seven discriminative regions, retains a global feature token, and classifies their concatenated 512-feature representation.

EVIDENCE: The strongest gains came from translation-matched training and translated-view inference, indicating that limited position invariance is valuable, while the current head remains load-bearing on exact spatial coordinates and compresses 3,136 activations to only 30. This tests the alternative assumption that content-addressed garment parts are more useful than a fixed-location bottleneck without the convolutional overhead that caused attention-gated and widened designs to time out.

<<<<<<< SEARCH
class ImageClassifier(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.GELU(),
            ResidualBlock(32, 32),
            nn.MaxPool2d(2),
            ResidualBlock(32, 64),
            nn.MaxPool2d(2),
            ResidualBlock(64, 64),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 30),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(30, 10),
        )

    def _forward_once(self, images: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(images))
=======
class AttentivePoolingHead(nn.Module):
    def __init__(self, channels: int = 64, heads: int = 7) -> None:
        super().__init__()
        self.attention = nn.Conv2d(channels, heads, kernel_size=1)
        self.classifier = nn.Sequential(
            nn.LayerNorm((heads + 1) * channels),
            nn.Linear((heads + 1) * channels, 160),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(160, 10),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        values = features.flatten(2)
        weights = self.attention(features).flatten(2).softmax(dim=-1)
        parts = torch.bmm(weights, values.transpose(1, 2))
        global_part = values.mean(dim=-1).unsqueeze(1)
        pooled = torch.cat((global_part, parts), dim=1)
        return self.classifier(pooled.flatten(1))


class ImageClassifier(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.GELU(),
            ResidualBlock(32, 32),
            nn.MaxPool2d(2),
            ResidualBlock(32, 64),
            nn.MaxPool2d(2),
            ResidualBlock(64, 64),
        )
        self.classifier = AttentivePoolingHead()

    def _forward_once(self, images: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(images))
>>>>>>> REPLACE