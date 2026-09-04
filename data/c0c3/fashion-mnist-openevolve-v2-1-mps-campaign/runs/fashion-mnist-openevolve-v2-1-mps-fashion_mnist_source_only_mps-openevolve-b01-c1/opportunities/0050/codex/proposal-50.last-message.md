MECHANISM: Multi-head spatial part pooling with geometric moments

HYPOTHESIS: Replacing fixed-coordinate flattening with six input-dependent attention pools will exceed 9,245 correct predictions by representing semantic garment parts and their spatial geometry while remaining robust to the translations already emphasized by crop ensembling.

INTENDED_EDIT: Replace the flattened 7×7 classifier with six learned spatial attention heads that pool backbone channels, append global statistics and five geometric moments per head, then classify the resulting descriptor with a 440-unit head; estimated parameters are 229,710.

EVIDENCE: Widening the flattened head regressed to 9,210 and appending global summaries reached only 9,236, showing that additional capacity and fixed pooled statistics were insufficient; the new head instead tests input-dependent spatial aggregation while preserving the 9,245-correct backbone and hierarchical objective.

<<<<<<< SEARCH
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(48 * 7 * 7, 80),
            nn.GELU(),
            nn.LayerNorm(80),
            nn.Linear(80, 10),
        )
=======
        self.attention = nn.Conv2d(48, 6, kernel_size=1)
        self.spatial_bias = nn.Parameter(torch.zeros(1, 6, 7, 7))

        axis = torch.linspace(-1.0, 1.0, 7)
        grid_y, grid_x = torch.meshgrid(axis, axis, indexing="ij")
        spatial_moments = torch.stack(
            (
                grid_x,
                grid_y,
                grid_x.square(),
                grid_y.square(),
                grid_x * grid_y,
            ),
            dim=-1,
        ).reshape(49, 5)
        self.register_buffer(
            "spatial_moments",
            spatial_moments,
            persistent=False,
        )

        self.classifier = nn.Sequential(
            nn.Linear(6 * 48 + 2 * 48 + 6 * 5, 440),
            nn.GELU(),
            nn.LayerNorm(440),
            nn.Linear(440, 10),
        )
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

        spatial_features = features.flatten(2).transpose(1, 2)
        attention_logits = self.attention(features) + self.spatial_bias
        attention_weights = F.softmax(
            attention_logits.flatten(2),
            dim=-1,
        )
        attended_features = torch.bmm(
            attention_weights,
            spatial_features,
        ).flatten(1)

        global_average = spatial_features.mean(dim=1)
        global_maximum = spatial_features.amax(dim=1)
        moments = self.spatial_moments.to(
            dtype=attention_weights.dtype,
        ).unsqueeze(0).expand(images.shape[0], -1, -1)
        attended_moments = torch.bmm(
            attention_weights,
            moments,
        ).flatten(1)

        descriptor = torch.cat(
            (
                attended_features,
                global_average,
                global_maximum,
                attended_moments,
            ),
            dim=1,
        )
        return self.classifier(descriptor)
>>>>>>> REPLACE