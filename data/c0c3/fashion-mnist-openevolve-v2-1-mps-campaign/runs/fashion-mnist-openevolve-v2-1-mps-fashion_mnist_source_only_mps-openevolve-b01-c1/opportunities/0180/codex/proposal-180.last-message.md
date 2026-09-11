MECHANISM: Position-aware global self-attention residual

HYPOTHESIS: A residual four-head attention layer over the 7×7 feature grid will exceed 9,267 correct predictions by learning content-dependent, garment-wide relationships while preserving the spatial layout required by the existing classifier.

INTENDED_EDIT: Insert a pre-normalized, position-aware self-attention block before the flattened classifier, with a learnable residual scale initialized conservatively; total learned parameters become 249,059.

EVIDENCE: Static dilated wider-context extraction regressed to 9,238 correct, while position-free pooling produced 9,253/9,249, indicating that spatial layout is load-bearing but fixed receptive-field expansion is insufficient. Dense input-conditioned attention tests a distinct global interaction mechanism without pooling away positions or replacing the successful flattened head.

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
        self.position_tokens = nn.Parameter(torch.zeros(1, 49, 56))
        self.context_norm = nn.LayerNorm(56)
        self.context_attention = nn.MultiheadAttention(
            56,
            num_heads=4,
            batch_first=True,
        )
        self.context_scale = nn.Parameter(torch.tensor(0.1))
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
        context_input = self.context_norm(tokens + self.position_tokens)
        context, _ = self.context_attention(
            context_input,
            context_input,
            context_input,
            need_weights=False,
        )
        tokens = tokens + self.context_scale * context
        features = tokens.transpose(1, 2).reshape(
            images.shape[0], 56, 7, 7
        )
        return self.classifier(features)
>>>>>>> REPLACE