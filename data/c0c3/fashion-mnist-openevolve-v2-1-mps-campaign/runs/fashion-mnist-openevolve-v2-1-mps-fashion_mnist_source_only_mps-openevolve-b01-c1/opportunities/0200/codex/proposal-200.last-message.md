MECHANISM: Position-aware nonlocal feature attention

HYPOTHESIS: Adding a lightweight self-attention residual over the 7×7 feature grid will exceed 9,267 correct predictions by learning content-dependent relationships between distant image parts while retaining the spatial layout required by the classifier.

INTENDED_EDIT: Insert a four-head, position-aware attention block between the convolutional residual features and flattening classifier, with a learnable per-channel residual scale; total parameters remain below 250,000.

EVIDENCE: Position-free pooling regressed, showing spatial layout is load-bearing, while uniform crop and rotation changes also regressed, showing indiscriminate invariance is harmful. Nonlocal attention preserves every spatial token and selectively changes interactions based on image content.

<<<<<<< SEARCH
        self.shortcut = nn.Sequential(
            nn.Conv2d(40, 56, kernel_size=1, bias=False),
            nn.BatchNorm2d(56),
        )
        self.classifier = nn.Sequential(
=======
        self.shortcut = nn.Sequential(
            nn.Conv2d(40, 56, kernel_size=1, bias=False),
            nn.BatchNorm2d(56),
        )
        self.position_embedding = nn.Parameter(torch.empty(1, 7 * 7, 56))
        nn.init.trunc_normal_(self.position_embedding, std=0.02)
        self.attention_norm = nn.LayerNorm(56)
        self.spatial_attention = nn.MultiheadAttention(
            56,
            num_heads=4,
            batch_first=True,
        )
        self.attention_scale = nn.Parameter(torch.full((56,), 0.1))
        self.classifier = nn.Sequential(
>>>>>>> REPLACE

<<<<<<< SEARCH
    def _forward_once(self, images: torch.Tensor) -> torch.Tensor:
        features = self.stem(images)
        features = F.gelu(self.residual(features) + self.shortcut(features))
        return self.classifier(features)
=======
    def _forward_once(self, images: torch.Tensor) -> torch.Tensor:
        features = self.stem(images)
        features = F.gelu(self.residual(features) + self.shortcut(features))
        tokens = features.flatten(2).transpose(1, 2)
        attention_input = self.attention_norm(
            tokens + self.position_embedding
        )
        attended = self.spatial_attention(
            attention_input,
            attention_input,
            attention_input,
            need_weights=False,
        )[0]
        tokens = tokens + self.attention_scale * attended
        features = tokens.transpose(1, 2).reshape_as(features)
        return self.classifier(features)
>>>>>>> REPLACE