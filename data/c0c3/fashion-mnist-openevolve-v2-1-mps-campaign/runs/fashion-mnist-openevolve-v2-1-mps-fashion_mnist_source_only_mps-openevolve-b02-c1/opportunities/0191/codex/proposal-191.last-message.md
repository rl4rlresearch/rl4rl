MECHANISM: Arithmetic posterior TTA aggregation

HYPOTHESIS: Averaging per-view class probabilities instead of unbounded logits will exceed 9,318 correct predictions by limiting the influence of confidently wrong shifted or flipped views.

INTENDED_EDIT: Convert each TTA view’s logits to probabilities before applying the existing center weights, then return temperature-scaled log-probabilities as valid ten-class logits.

EVIDENCE: Label smoothing improved correctness from 9,311 to 9,318, indicating that reducing overconfident decisions is beneficial; probability-space aggregation applies the same principle specifically to disagreements within the existing multi-view ensemble.

<<<<<<< SEARCH
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
=======
        ensemble = logits.softmax(dim=1) * 2.25
        for view in views[1:]:
            view_logits = self.classifier(self.features(view))
            ensemble = ensemble + view_logits.softmax(dim=1)
        for view_index, view in enumerate(views):
            flipped_logits = self.classifier(self.features(view.flip(-1)))
            flipped_probabilities = flipped_logits.softmax(dim=1)
            if view_index == 0:
                flipped_probabilities = flipped_probabilities * 2.25
            ensemble = ensemble + flipped_probabilities
        probabilities = (ensemble / 12.5).clamp_min(1e-8)
        return probabilities.log() / 0.9350
>>>>>>> REPLACE