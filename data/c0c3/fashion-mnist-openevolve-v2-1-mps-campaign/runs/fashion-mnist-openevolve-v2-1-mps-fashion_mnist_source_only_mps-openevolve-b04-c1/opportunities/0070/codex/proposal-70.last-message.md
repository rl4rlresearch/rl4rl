MECHANISM: Class-token relational reasoning over spatial feature tokens

HYPOTHESIS: Replacing fixed mirrored-feature fusion and a 48-template flattened MLP with two self-attention blocks over 7×7 local tokens will exceed 9,328 correct predictions by learning content-dependent relationships among image parts while reducing redundant paired-view computation.

INTENDED_EDIT: Remove mandatory mirrored-view processing and the flattened classifier; project the convolutional map into 49 tokens, add positional information and a learned class token, and classify through two pre-normalized self-attention blocks.

EVIDENCE: The fused nonlinear classifier reached 9,316 correct and a spatial evidence readout reached 9,307, showing spatial evidence is useful, while subsequent gains to 9,328 came from regularization rather than representational changes. This patch challenges the load-bearing assumption that static global templates over hand-symmetrized features are sufficient, using content-dependent spatial interactions instead.

<<<<<<< SEARCH
BATCH_SIZE = 64
GRAD_CLIP_NORM = 1.0


class ImageClassifier(nn.Module):
=======
BATCH_SIZE = 64
GRAD_CLIP_NORM = 1.0


class TokenBlock(nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(width)
        self.attention = nn.MultiheadAttention(
            width,
            num_heads=4,
            batch_first=True,
        )
        self.norm2 = nn.LayerNorm(width)
        self.mlp = nn.Sequential(
            nn.Linear(width, 2 * width),
            nn.GELU(),
            nn.Linear(2 * width, width),
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        normalized = self.norm1(tokens)
        attended, _ = self.attention(
            normalized,
            normalized,
            normalized,
            need_weights=False,
        )
        tokens = tokens + attended
        return tokens + self.mlp(self.norm2(tokens))


class ImageClassifier(nn.Module):
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.pool = nn.MaxPool2d(2)
        self.view_fusion = nn.Conv2d(
            128, 64, kernel_size=1, bias=False
        )
        with torch.no_grad():
            self.view_fusion.weight.zero_()
            self.view_fusion.weight[:, :64, 0, 0].copy_(torch.eye(64))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 48),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(48, 10),
        )
=======
        self.pool = nn.MaxPool2d(2)
        token_width = 96
        self.token_projection = nn.Conv2d(
            64, token_width, kernel_size=1, bias=False
        )
        self.class_token = nn.Parameter(torch.zeros(1, 1, token_width))
        self.position_embedding = nn.Parameter(
            torch.empty(1, 50, token_width)
        )
        nn.init.trunc_normal_(self.position_embedding, std=0.02)
        self.token_blocks = nn.ModuleList(
            TokenBlock(token_width) for _ in range(2)
        )
        self.final_norm = nn.LayerNorm(token_width)
        self.token_dropout = nn.Dropout(0.15)
        self.head = nn.Linear(token_width, 10)
>>>>>>> REPLACE

<<<<<<< SEARCH
    def _classify_views(
        self,
        features: torch.Tensor,
        flipped_features: torch.Tensor,
    ) -> torch.Tensor:
        invariant = 0.5 * (features + flipped_features)
        disagreement = torch.abs(features - flipped_features)
        fused = self.view_fusion(
            torch.cat((invariant, disagreement), dim=1)
        )
        return self.classifier(fused)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        if self.training:
            batch_size = images.shape[0]
            paired_images = torch.cat(
                (images, torch.flip(images, dims=(-1,))),
                dim=0,
            )
            paired_features = self._forward_features(paired_images)
            features = paired_features[:batch_size]
            flipped_features = paired_features[batch_size:]
        else:
            features = self._forward_features(images)
            flipped_features = self._forward_features(
                torch.flip(images, dims=(-1,))
            )

        logits = self._classify_views(features, flipped_features)
        if self.training:
            return logits
        return 1.2112 * logits
=======
    def _forward_logits(self, images: torch.Tensor) -> torch.Tensor:
        features = self.token_projection(self._forward_features(images))
        tokens = features.flatten(2).transpose(1, 2)
        class_token = self.class_token.expand(tokens.shape[0], -1, -1)
        tokens = torch.cat((class_token, tokens), dim=1)
        tokens = tokens + self.position_embedding
        for block in self.token_blocks:
            tokens = block(tokens)
        representation = self.final_norm(tokens[:, 0])
        return self.head(self.token_dropout(representation))

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        logits = self._forward_logits(images)
        if self.training:
            return logits
        return 1.2112 * logits
>>>>>>> REPLACE

<<<<<<< SEARCH
    model.classifier[3].p = 0.15 * (1.0 - dropout_decay)
=======
    model.token_dropout.p = 0.15 * (1.0 - dropout_decay)
>>>>>>> REPLACE