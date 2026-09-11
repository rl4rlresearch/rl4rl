MECHANISM: Canonical-view-biased flip logit ensemble

HYPOTHESIS: Weighting the canonical validation image slightly more than its synthetic horizontal flip will increase validation_correct above 9,359 while retaining most of the flip ensemble’s benefit.

INTENDED_EDIT: Change evaluation aggregation from an equal average to 55% original-image logits and 45% flipped-image logits, retaining the best-known calibration scale.

EVIDENCE: Adding weaker translated evaluation views reduced validation_correct from 9,359 to 9,348, showing that equal weighting of transformed views can hurt; a modest canonical-view bias tests view reliability without adding computation or altering training.

<<<<<<< SEARCH
        flipped_logits = self.classifier(self.features(images.flip(-1)))
        return 1.226016 * 0.5 * (logits + flipped_logits)
=======
        flipped_logits = self.classifier(self.features(images.flip(-1)))
        return 1.226016 * (0.55 * logits + 0.45 * flipped_logits)
>>>>>>> REPLACE