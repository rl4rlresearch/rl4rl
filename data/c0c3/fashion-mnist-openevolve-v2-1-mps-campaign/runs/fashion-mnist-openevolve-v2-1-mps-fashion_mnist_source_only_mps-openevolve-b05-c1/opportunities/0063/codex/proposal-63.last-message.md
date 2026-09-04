MECHANISM: Global-context bottleneck recalibration

HYPOTHESIS: Recalibrating the 30-dimensional spatial embedding from normalized global channel means will exceed 9,344 correct predictions while finishing within the time limit by retaining input-conditioned multiplicative gating without backpropagating a gate across the full feature map.

INTENDED_EDIT: Split the spatial classifier into projection and output layers, then add an identity-initialized 64→8→30 context gate that reuses the statistics head’s normalized means and modulates only the compact embedding; parameters increase to 248,208.

EVIDENCE: Full-map squeeze-and-excitation reported 9,345 correct, the only representation change above the current 9,344, but took 79.6 seconds, and its narrower successor also timed out. Moving recalibration after the existing bottleneck removes the expensive feature-map multiplication while directly testing the promising global-context signal.

<<<<<<< SEARCH
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 30),
            nn.BatchNorm1d(30),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(30, 10),
        )
        self.statistics_norm = nn.BatchNorm1d(64 * 3)
        self.statistics_head = nn.Linear(64 * 3, 10)
        nn.init.zeros_(self.statistics_head.weight)
        nn.init.zeros_(self.statistics_head.bias)
=======
        self.spatial_projection = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 30),
            nn.BatchNorm1d(30),
            nn.GELU(),
            nn.Dropout(0.1),
        )
        self.classifier = nn.Linear(30, 10)
        self.statistics_norm = nn.BatchNorm1d(64 * 3)
        self.statistics_head = nn.Linear(64 * 3, 10)
        nn.init.zeros_(self.statistics_head.weight)
        nn.init.zeros_(self.statistics_head.bias)
        self.context_gate = nn.Sequential(
            nn.Linear(64, 8),
            nn.GELU(),
            nn.Linear(8, 30),
        )
        nn.init.zeros_(self.context_gate[-1].weight)
        nn.init.zeros_(self.context_gate[-1].bias)
>>>>>>> REPLACE

<<<<<<< SEARCH
        residual_logits = self.statistics_head(
            self.statistics_norm(statistics)
        )
        return self.classifier(feature_map) + residual_logits
=======
        normalized_statistics = self.statistics_norm(statistics)
        residual_logits = self.statistics_head(normalized_statistics)
        spatial_embedding = self.spatial_projection(feature_map)
        context_scale = 1.0 + torch.tanh(
            self.context_gate(normalized_statistics[:, :64])
        )
        spatial_logits = self.classifier(
            spatial_embedding * context_scale
        )
        return spatial_logits + residual_logits
>>>>>>> REPLACE