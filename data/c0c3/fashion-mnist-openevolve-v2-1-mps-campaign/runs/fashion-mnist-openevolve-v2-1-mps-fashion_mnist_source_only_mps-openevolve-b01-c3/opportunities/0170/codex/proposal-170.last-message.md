MECHANISM: Content-adaptive saliency pooling

HYPOTHESIS: Replacing fixed global-average pooling with a zero-initialized learned spatial attention pool, while using the verified 5% consistency strength for every pair, will exceed 9,318 correct predictions by learning which image regions should contribute to each representation.

INTENDED_EDIT: Preserve the successful backbone, max-pooled evidence, paired training, and inference ensemble; replace uniform mean aggregation with learned attention initialized to identical behavior, slightly narrow the classifier to remain below 250,000 parameters, and restore equal-strength cross-offset consistency.

EVIDENCE: Equal 5% flip-offset consistency produced the best result at 9,318 correct, while further orbit-loss variations timed out and learned reflection fusion fell to 9,280. Those designs retain or manipulate fixed spatial aggregation; this patch instead challenges the load-bearing assumption that every spatial location should contribute equally to the global feature vector.

<<<<<<< SEARCH
        self.late_context = MultiScaleBlock(96)
        self.late_refinement = SpatialRefinement(96)
        self.classifier = nn.Sequential(
            nn.LayerNorm(192),
            nn.Linear(192, 61),
            nn.GELU(),
            nn.Dropout(0.10),
            nn.Linear(61, 10),
        )
=======
        self.late_context = MultiScaleBlock(96)
        self.late_refinement = SpatialRefinement(96)
        self.spatial_attention = nn.Conv2d(96, 1, kernel_size=1)
        nn.init.zeros_(self.spatial_attention.weight)
        nn.init.zeros_(self.spatial_attention.bias)
        self.classifier = nn.Sequential(
            nn.LayerNorm(192),
            nn.Linear(192, 60),
            nn.GELU(),
            nn.Dropout(0.10),
            nn.Linear(60, 10),
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
        features = self.late_context(features)
        features = self.late_refinement(features)
        mean_features = F.adaptive_avg_pool2d(features, 1).flatten(1)
        peak_features = F.adaptive_max_pool2d(features, 1).flatten(1)
        return self.classifier(torch.cat((mean_features, peak_features), dim=1))
=======
        features = self.late_context(features)
        features = self.late_refinement(features)
        flattened_features = features.flatten(2)
        attention_weights = F.softmax(
            self.spatial_attention(features).flatten(2),
            dim=-1,
        )
        attended_features = (
            flattened_features * attention_weights
        ).sum(dim=-1)
        peak_features = F.adaptive_max_pool2d(features, 1).flatten(1)
        return self.classifier(
            torch.cat((attended_features, peak_features), dim=1)
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
    consistency_weights = torch.where(
        cross_offset_mask,
        consistency_loss.new_tensor(0.025),
        consistency_loss.new_tensor(0.05),
    )
=======
    consistency_weights = torch.where(
        cross_offset_mask,
        consistency_loss.new_tensor(0.05),
        consistency_loss.new_tensor(0.05),
    )
>>>>>>> REPLACE