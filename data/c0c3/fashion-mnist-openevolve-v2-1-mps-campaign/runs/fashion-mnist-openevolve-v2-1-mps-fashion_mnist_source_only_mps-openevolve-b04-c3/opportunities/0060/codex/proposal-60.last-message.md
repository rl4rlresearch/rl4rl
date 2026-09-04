MECHANISM: Multi-query positional part-attention pooling

HYPOTHESIS: Replacing the fixed 38-unit flattened bottleneck with eight input-dependent spatial attention pools and a 228-unit fusion layer will exceed 9,243 correct predictions by retaining multiple discriminative part descriptors while learning where to gather them.

INTENDED_EDIT: Replace the parameter-dominant flattened classifier with learned positional attention pooling, restore the strongest verified 0.60-order flip ensemble, and restore the verified 12.5% cosine learning-rate floor; the resulting model has 249,342 learned parameters.

EVIDENCE: The best design reached 9,243 correct with the 0.60-order ensemble, but all available designs share a 179,256-parameter flatten-to-38 head. The failed global-average shortcut pooled the same shallow features into one first-order descriptor; eight learned, position-aware pools instead preserve complementary regions and fuse them through a substantially wider learned representation.

<<<<<<< SEARCH
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(96 * 7 * 7, 38),
            nn.LayerNorm(38),
            nn.GELU(),
            nn.Dropout(p=0.1),
            nn.Linear(38, 10),
        )

    def _classify(self, images: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.stem(images))

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        logits = self._classify(images)
        if not self.training:
            flipped_logits = self._classify(images.flip(-1))
            logits = 0.5 * (logits + flipped_logits)
        return logits
=======
        self.position = nn.Parameter(torch.zeros(1, 96, 7, 7))
        self.attention = nn.Conv2d(96, 8, kernel_size=1)
        self.classifier = nn.Sequential(
            nn.Linear(8 * 96, 228),
            nn.LayerNorm(228),
            nn.GELU(),
            nn.Dropout(p=0.1),
            nn.Linear(228, 10),
        )

    def _classify(self, images: torch.Tensor) -> torch.Tensor:
        features = self.stem(images) + self.position
        weights = F.softmax(self.attention(features).flatten(2), dim=-1)
        values = features.flatten(2).transpose(1, 2)
        pooled = torch.bmm(weights, values).flatten(1)
        return self.classifier(pooled)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        logits = self._classify(images)
        if not self.training:
            flipped_logits = self._classify(images.flip(-1))
            power = 0.60
            logits = (
                torch.logaddexp(
                    power * F.log_softmax(logits, dim=1),
                    power * F.log_softmax(flipped_logits, dim=1),
                )
                - math.log(2.0)
            ) / power
        return logits
>>>>>>> REPLACE

<<<<<<< SEARCH
    multiplier = 0.075 + 0.925 * 0.5 * (1.0 + math.cos(math.pi * progress))
=======
    multiplier = 0.125 + 0.875 * 0.5 * (1.0 + math.cos(math.pi * progress))
>>>>>>> REPLACE