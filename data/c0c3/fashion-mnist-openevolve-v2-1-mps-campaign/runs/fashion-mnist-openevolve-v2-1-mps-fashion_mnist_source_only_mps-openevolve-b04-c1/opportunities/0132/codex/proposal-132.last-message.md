MECHANISM: Annealed per-view auxiliary supervision

HYPOTHESIS: Training each mirrored feature view to remain independently class-discriminative, then decaying that auxiliary objective during EMA collection, will exceed 9,328 correct predictions without changing the verified inference pathway.

INTENDED_EDIT: Add training-only classifier losses for the original and mirrored feature tensors, weighted at 0.10 through the first half and annealed to zero during the second half.

EVIDENCE: Disagreement-only fusion fell to 9,290 and an invariant residual skip fell to 9,311, indicating that altering the successful inference fusion is harmful; auxiliary supervision instead strengthens its input features while preserving inference exactly.

<<<<<<< SEARCH
        logits = self._classify_views(features, flipped_features)
        if self.training:
            return logits
        return 1.2112 * logits
=======
        logits = self._classify_views(features, flipped_features)
        if self.training:
            self.auxiliary_view_logits = (
                self.classifier(features),
                self.classifier(flipped_features),
            )
            return logits
        return 1.2112 * logits
>>>>>>> REPLACE

<<<<<<< SEARCH
    logits = model(images)
    label_smoothing = 0.02 * (1.0 - dropout_decay)
    return F.cross_entropy(
        logits,
        labels,
        label_smoothing=label_smoothing,
    )
=======
    logits = model(images)
    label_smoothing = 0.02 * (1.0 - dropout_decay)
    main_loss = F.cross_entropy(
        logits,
        labels,
        label_smoothing=label_smoothing,
    )
    original_logits, flipped_logits = model.auxiliary_view_logits
    auxiliary_loss = 0.5 * (
        F.cross_entropy(
            original_logits,
            labels,
            label_smoothing=label_smoothing,
        )
        + F.cross_entropy(
            flipped_logits,
            labels,
            label_smoothing=label_smoothing,
        )
    )
    auxiliary_weight = 0.10 * (1.0 - dropout_decay)
    return main_loss + auxiliary_weight * auxiliary_loss
>>>>>>> REPLACE