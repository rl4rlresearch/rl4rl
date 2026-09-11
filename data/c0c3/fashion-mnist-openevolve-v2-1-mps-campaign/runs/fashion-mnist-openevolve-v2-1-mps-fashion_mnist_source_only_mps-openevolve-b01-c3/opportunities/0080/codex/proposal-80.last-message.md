MECHANISM: Channel-adaptive smooth-extremum feature pooling

HYPOTHESIS: Replacing hard global maxima with learned per-channel log-mean-exp pooling will exceed 9,287 correct predictions by preserving distributed detector evidence while suppressing isolated activation outliers.

INTENDED_EDIT: Challenge the assumption that each channel is best summarized by a single maximum; learn each channel’s pooling sharpness, slightly narrow the classifier to remain below 250,000 parameters, and retain the strongest linear-recency averaging with 1.125 logit scaling.

EVIDENCE: Uniform ten-view pooling reached 9,287 correct, whereas trimming view evidence fell to 9,276; this favors retaining all views while addressing outlier sensitivity inside the representation, before view aggregation.

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
        self.pool_temperature = nn.Parameter(torch.ones(96))
        self.classifier = nn.Sequential(
            nn.LayerNorm(192),
            nn.Linear(192, 60),
            nn.GELU(),
            nn.Dropout(0.10),
            nn.Linear(60, 10),
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
        mean_features = F.adaptive_avg_pool2d(features, 1).flatten(1)
        peak_features = F.adaptive_max_pool2d(features, 1).flatten(1)
        return self.classifier(torch.cat((mean_features, peak_features), dim=1))
=======
        flat_features = features.flatten(2)
        mean_features = flat_features.mean(dim=2)
        temperature = (
            0.25 + F.softplus(self.pool_temperature)
        )[None, :, None]
        smooth_peak_features = (
            torch.logsumexp(flat_features * temperature, dim=2)
            - math.log(flat_features.shape[2])
        ) / temperature.squeeze(2)
        pooled_features = torch.cat(
            (mean_features, smooth_peak_features),
            dim=1,
        )
        return self.classifier(pooled_features)
>>>>>>> REPLACE

<<<<<<< SEARCH
        return view_logits.mean(dim=(0, 1))
=======
        return 1.125 * view_logits.mean(dim=(0, 1))
>>>>>>> REPLACE

<<<<<<< SEARCH
        recency_power = 0.5
=======
        recency_power = 1.0
>>>>>>> REPLACE