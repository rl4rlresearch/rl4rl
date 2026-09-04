MECHANISM: Zero-initialized global-context logit shortcut

HYPOTHESIS: Adding a complementary global-average classifier to the verified 12.5%-floor design will exceed 9,240 correct predictions without materially increasing runtime.

INTENDED_EDIT: Restore logit-space flip ensembling and the 12.5% cosine floor, then add a zero-initialized 970-parameter global-pooled head alongside the fixed-position classifier.

EVIDENCE: The fixed flatten-head design reached 9,240, while widening its bottleneck fell to 9,223 and more computationally intensive spatial heads timed out; a cheap pooled shortcut adds translation-tolerant evidence without disrupting the proven classifier at initialization.

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
=======
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(96 * 7 * 7, 38),
            nn.LayerNorm(38),
            nn.GELU(),
            nn.Dropout(p=0.1),
            nn.Linear(38, 10),
        )
        self.global_classifier = nn.Linear(96, 10)
        nn.init.zeros_(self.global_classifier.weight)
        nn.init.zeros_(self.global_classifier.bias)

    def _classify(self, images: torch.Tensor) -> torch.Tensor:
        features = self.stem(images)
        global_features = features.mean(dim=(-2, -1))
        return self.classifier(features) + self.global_classifier(global_features)
>>>>>>> REPLACE

<<<<<<< SEARCH
            logits = torch.logaddexp(
                F.log_softmax(logits, dim=-1),
                F.log_softmax(flipped_logits, dim=-1),
            ) - math.log(2.0)
=======
            logits = 0.5 * (logits + flipped_logits)
>>>>>>> REPLACE

<<<<<<< SEARCH
    multiplier = 0.5 * (1.0 + math.cos(math.pi * progress))
=======
    multiplier = 0.125 + 0.875 * 0.5 * (1.0 + math.cos(math.pi * progress))
>>>>>>> REPLACE