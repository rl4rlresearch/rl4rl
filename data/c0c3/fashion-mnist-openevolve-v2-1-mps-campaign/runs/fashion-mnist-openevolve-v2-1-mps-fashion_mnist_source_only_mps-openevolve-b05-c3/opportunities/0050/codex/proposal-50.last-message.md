MECHANISM: Hybrid spatial-grid and global-distribution head

HYPOTHESIS: Combining the retained 3×3 max-pooled spatial grid with pre-pool channel averages will exceed 9,172 correct predictions by preserving position-sensitive evidence while recovering diffuse texture and boundary information discarded by the final max pool.

INTENDED_EDIT: Replace the assumption that final max-pooled activations alone contain all useful evidence with a learned head that concatenates local maxima and global channel means; reduce its hidden width to 47 to stay below 250,000 parameters, and use the best verified 19/32 curriculum.

EVIDENCE: The 19/32 curriculum achieved 9,172 correct, while widening the existing head fell to 9,164 and attention pooling reportedly fell to 9,103. This motivates adding complementary global evidence without discarding the validated positional representation or materially increasing convolutional computation.

<<<<<<< SEARCH
            if index in (1, 3, 5):
                layers.append(nn.MaxPool2d(2))
        self.features = nn.Sequential(*layers)
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
            if index in (1, 3):
                layers.append(nn.MaxPool2d(2))
        self.features = nn.Sequential(*layers)
        self.classifier = nn.Sequential(
            nn.Dropout(0.15),
            nn.Linear(96 * (3 * 3 + 1), 47),
            nn.SiLU(inplace=True),
            nn.Dropout(0.10),
            nn.Linear(47, 10),
        )

    def _forward_once(self, images: torch.Tensor) -> torch.Tensor:
        features = self.features(images)
        spatial_grid = F.max_pool2d(features, kernel_size=2).flatten(1)
        global_context = F.adaptive_avg_pool2d(features, 1).flatten(1)
        representation = torch.cat((spatial_grid, global_context), dim=1)
        return self.classifier(representation)
>>>>>>> REPLACE

<<<<<<< SEARCH
    if step * 64 < total_steps * 39:
=======
    if step * 32 < total_steps * 19:
>>>>>>> REPLACE