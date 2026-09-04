MECHANISM: Spatial-template and global-moment logit fusion

HYPOTHESIS: Adding a zero-initialized mean-and-standard-deviation prediction branch to the qualified 0.0325-buffer-EMA design will exceed 9,206 correct predictions by complementing coordinate-sensitive spatial templates with translation-stable channel-distribution evidence.

INTENDED_EDIT: Replace the old assumption that flattened 3×3 features alone should determine each class with an additive classifier that also learns from global first- and second-order feature moments; retain the strongest qualified EMA setting.

EVIDENCE: Reference Design 1 achieved 9,206 correct, while repeated EMA refinements plateaued at 9,205–9,206; the residual model’s advantage over the plain network indicates that a genuinely different learned representation mechanism is more promising than further coefficient tuning.

<<<<<<< SEARCH
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.15),
            nn.Linear(96 * 3 * 3, 48),
            nn.SiLU(inplace=True),
            nn.Dropout(0.10),
            nn.Linear(48, 10),
        )

    def _forward_once(self, images: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(images))
=======
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.15),
            nn.Linear(96 * 3 * 3, 48),
            nn.SiLU(inplace=True),
            nn.Dropout(0.10),
            nn.Linear(48, 10),
        )
        self.moment_classifier = nn.Linear(96 * 2, 10)
        nn.init.zeros_(self.moment_classifier.weight)
        nn.init.zeros_(self.moment_classifier.bias)

    def _forward_once(self, images: torch.Tensor) -> torch.Tensor:
        features = self.features(images)
        channel_mean = features.mean(dim=(2, 3))
        channel_std = features.var(
            dim=(2, 3), unbiased=False
        ).add(1e-6).sqrt()
        moments = torch.cat((channel_mean, channel_std), dim=1)
        return self.classifier(features) + self.moment_classifier(moments)
>>>>>>> REPLACE

<<<<<<< SEARCH
                    ema_rate = 0.04 if is_buffer else 0.015
=======
                    ema_rate = 0.0325 if is_buffer else 0.015
>>>>>>> REPLACE