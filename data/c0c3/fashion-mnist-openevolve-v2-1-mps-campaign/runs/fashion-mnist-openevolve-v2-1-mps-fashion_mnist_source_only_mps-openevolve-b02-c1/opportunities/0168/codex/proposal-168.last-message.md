MECHANISM: Arithmetic probability TTA aggregation

HYPOTHESIS: Averaging per-view class probabilities instead of logits will suppress isolated overconfident TTA errors and exceed 9,311 correct predictions while preserving temperature calibration.

INTENDED_EDIT: Replace evaluation-time weighted logit averaging with the same 12-weight arithmetic probability ensemble, converted back to calibrated logits.

EVIDENCE: Temperature tuning plateaued at exactly 9,311 correct, so progress requires changed argmax decisions; this isolates the ensemble aggregation rule without altering training cost, parameters, or learned weights.

<<<<<<< SEARCH
        ensemble = logits * 2.0
        for view in views[1:]:
            view_logits = self.classifier(self.features(view))
            ensemble = ensemble + view_logits
        for view_index, view in enumerate(views):
            flipped_logits = self.classifier(self.features(view.flip(-1)))
            if view_index == 0:
                flipped_logits = flipped_logits * 2.0
            ensemble = ensemble + flipped_logits
        return ensemble / (12.0 * 0.9350)
=======
        ensemble = logits.softmax(dim=1) * 2.0
        for view in views[1:]:
            view_logits = self.classifier(self.features(view))
            ensemble = ensemble + view_logits.softmax(dim=1)
        for view_index, view in enumerate(views):
            flipped_logits = self.classifier(self.features(view.flip(-1)))
            flipped_probabilities = flipped_logits.softmax(dim=1)
            if view_index == 0:
                flipped_probabilities = flipped_probabilities * 2.0
            ensemble = ensemble + flipped_probabilities
        probabilities = (ensemble / 12.0).clamp_min(1e-12)
        return probabilities.log() / 0.9350
>>>>>>> REPLACE