MECHANISM: Global-context channel recalibration

HYPOTHESIS: Replacing the fixed channel mixture with input-conditioned multiplicative gating will exceed 9,342 correct predictions because global context can suppress irrelevant feature channels before both the spatial classifier and statistics head compute logits.

INTENDED_EDIT: Add a zero-initialized squeeze-and-excitation gate over the trunk’s global channel means and use it to recalibrate the feature map; the model begins exactly equivalent to the current design and has 249,546 learned parameters.

EVIDENCE: The global-statistics bypass improved correctness from 9,314 to 9,334, showing that pooled channel context contains useful information, while EMA raised the unchanged representation to 9,342. This challenges the shared assumption that global context should act only as an additive final-logit correction by instead using it to alter the image representation itself.

<<<<<<< SEARCH
        self.classifier = nn.Sequential(
=======
        self.context_gate = nn.Sequential(
            nn.Linear(64, 16),
            nn.GELU(),
            nn.Linear(16, 64),
        )
        nn.init.zeros_(self.context_gate[-1].weight)
        nn.init.zeros_(self.context_gate[-1].bias)
        self.classifier = nn.Sequential(
>>>>>>> REPLACE

<<<<<<< SEARCH
        feature_map = self.features(images)
        spatial_mean = feature_map.mean(dim=(2, 3))
        spatial_std = feature_map.var(
            dim=(2, 3), unbiased=False
        ).add(1e-4).sqrt()
=======
        feature_map = self.features(images)
        ungated_mean = feature_map.mean(dim=(2, 3))
        channel_scale = 2.0 * torch.sigmoid(
            self.context_gate(ungated_mean)
        )
        feature_map = feature_map * channel_scale[:, :, None, None]
        spatial_mean = ungated_mean * channel_scale
        spatial_std = feature_map.var(
            dim=(2, 3), unbiased=False
        ).add(1e-4).sqrt()
>>>>>>> REPLACE