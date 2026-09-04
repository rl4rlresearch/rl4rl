MECHANISM: Center-prior weighted logit TTA

HYPOTHESIS: Increasing each centered view’s weight from 2 to 3 will exceed 9,311 correct predictions by reducing the influence of padding-affected shifted views while preserving transformation ensembling.

INTENDED_EDIT: Give the original and horizontally flipped centered views weight 3, and renormalize the ensemble’s temperature-scaled denominator from 12 to 14.

EVIDENCE: Weighted-logit TTA improved the best result from 9,310 to 9,311 correct, demonstrating that aggregation changes borderline decisions; center reweighting tests view reliability without altering training, parameters, or forward count.

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
        return ensemble / (12.0 * 0.912)
=======
        ensemble = logits * 3.0
        for view in views[1:]:
            view_logits = self.classifier(self.features(view))
            ensemble = ensemble + view_logits
        for view_index, view in enumerate(views):
            flipped_logits = self.classifier(self.features(view.flip(-1)))
            if view_index == 0:
                flipped_logits = flipped_logits * 3.0
            ensemble = ensemble + flipped_logits
        return ensemble / (14.0 * 0.912)
>>>>>>> REPLACE