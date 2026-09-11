MECHANISM: Class-conditioned spatial attention residual

HYPOTHESIS: A class-specific attention head that learns where to pool feature evidence will exceed 9,348 correct predictions, unlike indiscriminate global pooling, while retaining the strong flattened spatial head.

INTENDED_EDIT: Replace one dense bottleneck unit with a zero-initialized class-conditioned attention/value pooling branch, add learned class-specific spatial priors, and restore the best-verified TTA weights.

EVIDENCE: TTA bisection saturated at 9,348 correct, while global pooling fell to 9,085, showing that spatial organization is load-bearing. The old assumption is that a single flattened bottleneck should compute every prediction; this patch preserves that path but adds content-dependent, class-specific spatial aggregation instead of discarding location indiscriminately.

<<<<<<< SEARCH
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 56),
            nn.GELU(),
            nn.Dropout(p=0.10),
            nn.Linear(56, 10),
        )

    def _forward_once(self, images: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(images))
=======
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 55),
            nn.GELU(),
            nn.Dropout(p=0.10),
            nn.Linear(55, 10),
        )
        self.attention_score = nn.Conv2d(64, 10, kernel_size=1)
        self.attention_value = nn.Conv2d(64, 10, kernel_size=1)
        self.attention_position = nn.Parameter(
            torch.zeros(1, 10, 7, 7)
        )
        nn.init.zeros_(self.attention_score.weight)
        nn.init.zeros_(self.attention_score.bias)
        nn.init.zeros_(self.attention_value.weight)
        nn.init.zeros_(self.attention_value.bias)

    def _forward_once(self, images: torch.Tensor) -> torch.Tensor:
        features = self.features(images)
        scores = self.attention_score(features)
        scores = scores + self.attention_position
        attention = F.softmax(scores.flatten(2), dim=-1)
        values = self.attention_value(features).flatten(2)
        attended_logits = (attention * values).sum(dim=-1)
        return self.classifier(features) + attended_logits
>>>>>>> REPLACE

<<<<<<< SEARCH
        crop_weights = (
            3.0,
            1.314453125,
            1.314453125,
            0.685546875,
            0.685546875,
        )
=======
        crop_weights = (
            3.0,
            1.3134765625,
            1.3134765625,
            0.6865234375,
            0.6865234375,
        )
>>>>>>> REPLACE