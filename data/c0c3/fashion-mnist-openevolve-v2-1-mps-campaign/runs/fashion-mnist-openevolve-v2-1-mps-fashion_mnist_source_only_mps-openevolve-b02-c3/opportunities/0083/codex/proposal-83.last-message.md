MECHANISM: Consensus-favoring power-mean test-time ensemble

HYPOTHESIS: A 0.75-order power mean with the verified center weight of 3.0 will exceed 9,344 correct predictions by suppressing classes supported by only a few augmented views while preserving the proven crop balance.

INTENDED_EDIT: Restore the best-observed center weight and replace arithmetic probability averaging with a conservative power mean between arithmetic and geometric aggregation.

EVIDENCE: The weight-3 arithmetic ensemble achieved the best observed 9,344 correct, while crop-weight changes moved correctness as low as 9,338; this shows augmented-view aggregation affects boundary decisions and motivates testing consensus sensitivity without changing training.

<<<<<<< SEARCH
        crop_weights = (2.5, 1.0, 1.0, 1.0, 1.0)
        for crop, weight in zip(crops, crop_weights):
            probabilities = probabilities + weight * F.softmax(
                self._forward_once(crop), dim=1
            )
            probabilities = probabilities + weight * F.softmax(
                self._forward_once(crop.flip(-1)), dim=1
            )
        probabilities = probabilities / (2.0 * sum(crop_weights))
        return probabilities.clamp_min(
            torch.finfo(probabilities.dtype).tiny
        ).log() / 0.75317
=======
        crop_weights = (3.0, 1.0, 1.0, 1.0, 1.0)
        ensemble_power = 0.75
        for crop, weight in zip(crops, crop_weights):
            probabilities = probabilities + weight * F.softmax(
                self._forward_once(crop), dim=1
            ).pow(ensemble_power)
            probabilities = probabilities + weight * F.softmax(
                self._forward_once(crop.flip(-1)), dim=1
            ).pow(ensemble_power)
        probabilities = probabilities / (2.0 * sum(crop_weights))
        return probabilities.clamp_min(
            torch.finfo(probabilities.dtype).tiny
        ).log() / (ensemble_power * 0.75317)
>>>>>>> REPLACE