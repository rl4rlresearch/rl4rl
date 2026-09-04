MECHANISM: Center-emphasized geometric TTA ensemble

HYPOTHESIS: Increasing both centered-view weights from 2 to 3 while retaining all shifted views will exceed 9,311 correct predictions by reducing decisions driven by border-discarding translated crops.

INTENDED_EDIT: Reweight the existing logit ensemble toward the two centered views and renormalize its calibrated denominator from 12 to 14.

EVIDENCE: Arithmetic probability aggregation reduced accuracy to 9,310, supporting retention of geometric logit aggregation; temperature tuning cannot change the 9,311 argmax plateau, so view weighting is the smallest isolated change capable of improving the primary objective.

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
        ensemble = logits * 3.0
        for view in views[1:]:
            view_logits = self.classifier(self.features(view))
            ensemble = ensemble + view_logits
        for view_index, view in enumerate(views):
            flipped_logits = self.classifier(self.features(view.flip(-1)))
            if view_index == 0:
                flipped_logits = flipped_logits * 3.0
            ensemble = ensemble + flipped_logits
        return ensemble / (14.0 * 0.9350)
>>>>>>> REPLACE