MECHANISM: Conservative center-emphasized geometric TTA

HYPOTHESIS: Raising both centered-view weights from 2.0 to 2.25 will change fewer borderline decisions than the unverified 2.5 trial while correcting enough crop-sensitive errors to exceed 9,311 correct predictions.

INTENDED_EDIT: Increase the original and horizontally flipped centered-view weights to 2.25 and renormalize the calibrated ensemble by its total weight of 12.5.

EVIDENCE: Probability averaging reduced correctness to 9,310, supporting continued logit aggregation; temperature calibration cannot change the 9,311 argmax plateau, while stronger center weighting timed out without contradictory metric evidence, motivating a smaller isolated reweighting.

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
        ensemble = logits * 2.25
        for view in views[1:]:
            view_logits = self.classifier(self.features(view))
            ensemble = ensemble + view_logits
        for view_index, view in enumerate(views):
            flipped_logits = self.classifier(self.features(view.flip(-1)))
            if view_index == 0:
                flipped_logits = flipped_logits * 2.25
            ensemble = ensemble + flipped_logits
        return ensemble / (12.5 * 0.9350)
>>>>>>> REPLACE