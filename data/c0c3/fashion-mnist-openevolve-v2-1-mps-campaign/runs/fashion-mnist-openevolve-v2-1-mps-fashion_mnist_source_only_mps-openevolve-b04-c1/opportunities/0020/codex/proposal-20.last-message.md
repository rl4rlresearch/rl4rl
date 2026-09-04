MECHANISM: Class-query global-attention readout

HYPOTHESIS: Replacing the shared flattened MLP with ten class-specific attention queries over low-resolution image tokens will exceed 9,286 correct predictions by letting each class aggregate different long-range spatial evidence.

INTENDED_EDIT: Retain the proven convolutional front end, mirrored-pair objective, optimizer, and EMA, but replace the flattened classifier with positional tokens, two lightweight global-attention layers, and one learned query/readout per class.

EVIDENCE: Deepening the existing max-pooled convolutional branches regressed from 9,212 to 9,196, while learned convolutional downsampling timed out. The 9,286-best design still assumes local features followed by a shared flattened representation; low-resolution class-query attention tests a different prediction mechanism without expensive high-resolution convolutions.

<<<<<<< SEARCH
class ImageClassifier(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.GELU(),
        )
        self.residual1 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
        )
        self.transition = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.GELU(),
        )
        self.residual2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
        )
        self.pool = nn.MaxPool2d(2)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 48),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(48, 10),
        )

    def _forward_view(self, images: torch.Tensor) -> torch.Tensor:
        features = self.stem(images)
        features = F.gelu(features + self.residual1(features))
        features = self.pool(features)
        features = self.transition(features)
        features = F.gelu(features + self.residual2(features))
        features = self.pool(features)
        return self.classifier(features)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        logits = self._forward_view(images)
        if self.training:
            return logits
        flipped_logits = self._forward_view(torch.flip(images, dims=(-1,)))
        return 0.5 * (logits + flipped_logits)
=======
class ImageClassifier(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.GELU(),
        )
        self.residual1 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
        )
        self.transition = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.GELU(),
        )
        self.residual2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
        )
        self.pool = nn.MaxPool2d(2)

        token_width = 80
        self.token_projection = nn.Sequential(
            nn.Conv2d(64, token_width, kernel_size=1, bias=False),
            nn.BatchNorm2d(token_width),
            nn.GELU(),
        )
        self.image_positions = nn.Parameter(
            torch.empty(1, 7 * 7, token_width)
        )
        self.class_tokens = nn.Parameter(
            torch.empty(1, 10, token_width)
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=token_width,
            nhead=4,
            dim_feedforward=2 * token_width,
            dropout=0.10,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.global_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=2,
        )
        self.output_norm = nn.LayerNorm(token_width)
        self.classifier_weight = nn.Parameter(
            torch.empty(10, token_width)
        )
        self.classifier_bias = nn.Parameter(torch.zeros(10))

        nn.init.trunc_normal_(self.image_positions, std=0.02)
        nn.init.trunc_normal_(self.class_tokens, std=0.02)
        nn.init.trunc_normal_(self.classifier_weight, std=0.02)

    def _forward_view(self, images: torch.Tensor) -> torch.Tensor:
        features = self.stem(images)
        features = F.gelu(features + self.residual1(features))
        features = self.pool(features)
        features = self.transition(features)
        features = F.gelu(features + self.residual2(features))
        features = self.pool(features)

        image_tokens = self.token_projection(features)
        image_tokens = image_tokens.flatten(2).transpose(1, 2)
        image_tokens = image_tokens + self.image_positions
        class_tokens = self.class_tokens.expand(
            images.shape[0], -1, -1
        )
        tokens = torch.cat((class_tokens, image_tokens), dim=1)
        tokens = self.global_encoder(tokens)
        class_features = self.output_norm(tokens[:, :10])
        return (
            class_features * self.classifier_weight.unsqueeze(0)
        ).sum(dim=-1) + self.classifier_bias

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        logits = self._forward_view(images)
        if self.training:
            return logits
        flipped_logits = self._forward_view(torch.flip(images, dims=(-1,)))
        return 0.5 * (logits + flipped_logits)
>>>>>>> REPLACE