MECHANISM: Calibrated probability-space test-time augmentation

HYPOTHESIS: Averaging calibrated probabilities across the ten evaluation views will exceed 9,360 correct predictions by reducing domination from unusually confident individual views.

INTENDED_EDIT: Replace weighted logit averaging with weighted probability averaging while preserving the verified center weight and temperature.

EVIDENCE: Architecture and BatchNorm-state changes reduced accuracy, while the prior probability-ensemble verification only timed out and produced no contrary evidence, making a clean retry the most informative ranking-focused change.

<<<<<<< SEARCH
        center_weight = 1.81732177734375
        logits = center_weight * self._forward_once(views[0])
        logits = logits + center_weight * self._forward_once(
            views[0].flip(-1)
        )
        for view in views[1:]:
            logits = logits + self._forward_once(view)
            logits = logits + self._forward_once(view.flip(-1))
        logits = logits / (
            2.0 * center_weight + 2.0 * (len(views) - 1)
        )
        return 1.0671112 * logits
=======
        center_weight = 1.81732177734375
        scale = 1.0671112
        probabilities = center_weight * F.softmax(
            scale * self._forward_once(views[0]), dim=1
        )
        probabilities = probabilities + center_weight * F.softmax(
            scale * self._forward_once(views[0].flip(-1)), dim=1
        )
        for view in views[1:]:
            probabilities = probabilities + F.softmax(
                scale * self._forward_once(view), dim=1
            )
            probabilities = probabilities + F.softmax(
                scale * self._forward_once(view.flip(-1)), dim=1
            )
        probabilities = probabilities / (
            2.0 * center_weight + 2.0 * (len(views) - 1)
        )
        return probabilities.log()
>>>>>>> REPLACE