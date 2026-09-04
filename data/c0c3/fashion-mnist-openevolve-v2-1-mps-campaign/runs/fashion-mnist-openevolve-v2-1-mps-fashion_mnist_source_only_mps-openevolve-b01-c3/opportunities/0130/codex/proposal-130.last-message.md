MECHANISM: Content-conditioned spatial attention pooling

HYPOTHESIS: A zero-initialized, translation-compatible saliency pool will exceed 9,287 correct predictions by learning image-dependent regions while retaining fixed mean pooling as its initial state.

INTENDED_EDIT: Replace global mean pooling with an 18-parameter spatial-attention-weighted mean, retain global maxima, and use the best verified agreement-conditioned calibration.

EVIDENCE: Static vertical pooling fell to 9,252 correct, suggesting absolute-position summaries are harmful; content-conditioned attention instead preserves shift compatibility and can remain uniform when spatial selection is unhelpful.

<<<<<<< SEARCH
        self.late_context = MultiScaleBlock(96)
        self.late_refinement = SpatialRefinement(96)
        self.classifier = nn.Sequential(
=======
        self.late_context = MultiScaleBlock(96)
        self.late_refinement = SpatialRefinement(96)
        self.spatial_attention = nn.Conv2d(
            2, 1, kernel_size=3, padding=1, bias=False
        )
        nn.init.zeros_(self.spatial_attention.weight)
        self.classifier = nn.Sequential(
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
        attention_inputs = torch.cat(
            (
                features.mean(dim=1, keepdim=True),
                features.amax(dim=1, keepdim=True),
            ),
            dim=1,
        )
        attention_weights = F.softmax(
            self.spatial_attention(attention_inputs).flatten(2),
            dim=-1,
        )
        mean_features = (
            features.flatten(2) * attention_weights
        ).sum(dim=-1)
        peak_features = F.adaptive_max_pool2d(features, 1).flatten(1)
        return self.classifier(torch.cat((mean_features, peak_features), dim=1))
>>>>>>> REPLACE

<<<<<<< SEARCH
        pooled_logits = (
            1.5578756246377452 * offset_logits[0] + offset_logits[1:].sum(dim=0)
        ) / 5.5578756246377452
        return 1.22775 * pooled_logits
=======
        pooled_logits = (
            1.5578022218542173 * offset_logits[0] + offset_logits[1:].sum(dim=0)
        ) / 5.5578022218542173
        pooled_predictions = pooled_logits.argmax(dim=-1)
        view_agreement = (
            view_logits.argmax(dim=-1) == pooled_predictions[None, None, :]
        ).float().mean(dim=(0, 1))
        calibration = 1.22775 * (0.85 + 0.15 * view_agreement)
        return calibration.unsqueeze(1) * pooled_logits
>>>>>>> REPLACE