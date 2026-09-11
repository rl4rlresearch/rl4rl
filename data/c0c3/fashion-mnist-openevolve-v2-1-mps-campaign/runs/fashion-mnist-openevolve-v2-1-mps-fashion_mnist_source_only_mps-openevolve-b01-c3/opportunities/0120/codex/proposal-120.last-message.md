MECHANISM: Learned vertical spatial-pyramid pooling

HYPOTHESIS: Two learned top/bottom feature summaries plus global peak features will exceed 9,287 correct predictions by preserving class-relevant vertical structure discarded by global mean pooling.

INTENDED_EDIT: Replace global mean pooling with a mirrored, end-to-end learned seven-row pooling kernel; feed both vertical summaries and global maxima to a resized bias-free hidden layer, totaling 249,995 learned parameters.

EVIDENCE: Center-view refinements plateaued at 9,287 correct and the 1.5583894441224401× probe lost a prediction, indicating evaluation aggregation is saturated. The load-bearing untested assumption is that spatially invariant mean/max statistics contain enough shape information; this patch instead learns coarse vertical organization while retaining the proven peak descriptor.

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
        self.vertical_pool_logits = nn.Parameter(
            torch.tensor((1.0, 1.0, 1.0, 1.0, -1.0, -1.0, -1.0))
        )
        self.classifier = nn.Sequential(
            nn.LayerNorm(288),
            nn.Linear(288, 41, bias=False),
            nn.GELU(),
            nn.Dropout(0.10),
            nn.Linear(41, 10),
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
        mean_features = F.adaptive_avg_pool2d(features, 1).flatten(1)
        peak_features = F.adaptive_max_pool2d(features, 1).flatten(1)
        return self.classifier(torch.cat((mean_features, peak_features), dim=1))
=======
        top_weights = self.vertical_pool_logits.softmax(dim=0)
        vertical_weights = torch.stack(
            (top_weights, top_weights.flip(0)),
            dim=1,
        )
        spatial_features = torch.matmul(
            features.mean(dim=3),
            vertical_weights,
        ).flatten(1)
        peak_features = F.adaptive_max_pool2d(features, 1).flatten(1)
        return self.classifier(
            torch.cat((spatial_features, peak_features), dim=1)
        )
>>>>>>> REPLACE