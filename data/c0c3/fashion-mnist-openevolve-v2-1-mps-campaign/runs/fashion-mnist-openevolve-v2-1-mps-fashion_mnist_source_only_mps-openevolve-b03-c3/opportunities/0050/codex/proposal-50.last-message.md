MECHANISM: Residual nonlocal token attention

HYPOTHESIS: Adding content-dependent interactions among all 49 spatial locations will exceed 9,240 correct predictions by capturing relationships between distant garment parts that the locally processed dense baseline cannot model nonlinearly.

INTENDED_EDIT: Replace the 48-unit dense bottleneck with a pre-normalized spatial self-attention block and a smaller 32-unit classifier, remaining under the parameter ceiling; also retain the strongest verified smoothing anneal.

EVIDENCE: The baseline reached 9,240 correct, while separable coordinate gating fell to 9,225, suggesting coarse axis summaries are harmful. This tests a distinct mechanism—pairwise content-conditioned spatial mixing—and builds on the verified zero-ending smoothing improvement.

<<<<<<< SEARCH
        self.refine = nn.Sequential(
            nn.Conv2d(
                80, 80, kernel_size=3, padding=1, groups=80, bias=False
            ),
            nn.BatchNorm2d(80),
            nn.GELU(),
            nn.Conv2d(80, 80, kernel_size=1, bias=False),
            nn.BatchNorm2d(80),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(80 * 7 * 7, 48),
            nn.LayerNorm(48),
            nn.GELU(),
            nn.Dropout(0.10),
            nn.Linear(48, 10),
        )
=======
        self.refine = nn.Sequential(
            nn.Conv2d(
                80, 80, kernel_size=3, padding=1, groups=80, bias=False
            ),
            nn.BatchNorm2d(80),
            nn.GELU(),
            nn.Conv2d(80, 80, kernel_size=1, bias=False),
            nn.BatchNorm2d(80),
        )
        self.position = nn.Parameter(torch.zeros(1, 7 * 7, 80))
        self.attention_norm = nn.LayerNorm(80)
        self.attention = nn.MultiheadAttention(
            80, num_heads=4, dropout=0.05, batch_first=True
        )
        self.context_norm = nn.LayerNorm(80)
        self.context_mlp = nn.Sequential(
            nn.Linear(80, 160),
            nn.GELU(),
            nn.Dropout(0.05),
            nn.Linear(160, 80),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(80 * 7 * 7, 32),
            nn.LayerNorm(32),
            nn.GELU(),
            nn.Dropout(0.10),
            nn.Linear(32, 10),
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
    def _forward_once(self, images: torch.Tensor) -> torch.Tensor:
        features = self.features(self._image_basis(images))
        features = F.gelu(features + self.refine(features))
        return self.classifier(features)
=======
    def _forward_once(self, images: torch.Tensor) -> torch.Tensor:
        features = self.features(self._image_basis(images))
        features = F.gelu(features + self.refine(features))
        tokens = features.flatten(2).transpose(1, 2) + self.position

        normalized = self.attention_norm(tokens)
        attended = self.attention(
            normalized, normalized, normalized, need_weights=False
        )[0]
        tokens = tokens + attended
        tokens = tokens + self.context_mlp(self.context_norm(tokens))
        return self.classifier(tokens)
>>>>>>> REPLACE

<<<<<<< SEARCH
    smoothing = 0.03 + 0.02 * math.cos(math.pi * progress)
=======
    smoothing = 0.02 * (1.0 + math.cos(math.pi * progress))
>>>>>>> REPLACE