MECHANISM: Confidence-favoring power-mean test-time ensemble

HYPOTHESIS: A 1.25-order power mean with the verified center weight of 3.0 will exceed 9,344 correct predictions by reversing the harmful consensus bias observed with the 0.75-order mean and giving stronger augmented-view evidence modestly more influence.

INTENDED_EDIT: Restore the best-observed center crop weight and replace arithmetic probability averaging with a conservative 1.25-order power mean.

EVIDENCE: The weight-3 arithmetic ensemble achieved 9,344 correct, while shifting toward consensus with a 0.75-order power mean reduced correctness to 9,343 and worsened cross-entropy; testing the opposite side of arithmetic is the most direct directional follow-up.

<<<<<<< SEARCH
        crop_weights = (2.75, 1.0, 1.0, 1.0, 1.0)
        for crop, weight in zip(crops, crop_weights):
            probabilities = probabilities + weight * F.softmax(
                self._forward_once(crop), dim=1
            )
            probabilities = probabilities + weight * F.softmax(
                self._forward_once(crop.flip(-1)), dim=1
            )
        probabilities = probabilities / (2.0 * sum(crop_weights))
=======
        crop_weights = (3.0, 1.0, 1.0, 1.0, 1.0)
        power = 1.25
        for crop, weight in zip(crops, crop_weights):
            probabilities = probabilities + weight * F.softmax(
                self._forward_once(crop), dim=1
            ).pow(power)
            probabilities = probabilities + weight * F.softmax(
                self._forward_once(crop.flip(-1)), dim=1
            ).pow(power)
        probabilities = (
            probabilities / (2.0 * sum(crop_weights))
        ).pow(1.0 / power)
>>>>>>> REPLACE