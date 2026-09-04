MECHANISM: Full-resolution convolutional token transformer

HYPOTHESIS: Replacing fixed 3×3 flattening with two layers of content-dependent global token interaction over the complete 7×7 feature map will exceed 9,311 correct validation predictions while remaining below 250,000 parameters.

INTENDED_EDIT: Remove the terminal max-pool and dense classifier, reduce terminal width to 88 channels, and classify a learned global token after two pre-normalized self-attention blocks; retain the proven training procedure and weighted logit TTA.

EVIDENCE: Expanding the fixed dense head improved accuracy only to 9,310 correct and logit TTA reached 9,311, while another fixed terminal-pooling scheme regressed to 9,234. This challenges the load-bearing assumption that static pooling and flattening adequately model spatial relationships by introducing input-dependent global interactions at 241,778 parameters.

<<<<<<< SEARCH
GRAD_CLIP_NORM = 1.0


class ImageClassifier(nn.Module):
=======
GRAD_CLIP_NORM = 1.0


class SpatialTransformerHead(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.class_token = nn.Parameter(torch.zeros(1, 1, channels))
        self.position = nn.Parameter(torch.zeros(1, 50, channels))
        layer = nn.TransformerEncoderLayer(
            d_model=channels,
            nhead=4,
            dim_feedforward=160,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            layer,
            num_layers=2,
            enable_nested_tensor=False,
        )
        self.norm = nn.LayerNorm(channels)
        self.output = nn.Linear(channels, 10)
        nn.init.trunc_normal_(self.class_token, std=0.02)
        nn.init.trunc_normal_(self.position, std=0.02)

    def forward(self, feature_map: torch.Tensor) -> torch.Tensor:
        tokens = feature_map.flatten(2).transpose(1, 2)
        class_token = self.class_token.expand(tokens.shape[0], -1, -1)
        tokens = torch.cat((class_token, tokens), dim=1)
        tokens = tokens + self.position[:, : tokens.shape[1]]
        tokens = self.encoder(tokens)
        return self.output(self.norm(tokens[:, 0]))


class ImageClassifier(nn.Module):
>>>>>>> REPLACE

<<<<<<< SEARCH
            nn.Conv2d(64, 96, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(96),
            nn.GELU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(96 * 3 * 3, 147),
            nn.BatchNorm1d(147),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(147, 10),
        )
=======
            nn.Conv2d(64, 88, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(88),
            nn.GELU(),
        )
        self.classifier = SpatialTransformerHead(88)
>>>>>>> REPLACE