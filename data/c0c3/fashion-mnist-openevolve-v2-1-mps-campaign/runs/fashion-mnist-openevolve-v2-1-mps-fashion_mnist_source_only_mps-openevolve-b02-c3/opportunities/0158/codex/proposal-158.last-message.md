MECHANISM: Hierarchical orientation consensus with one-ULP sharpening

HYPOTHESIS: The verified hierarchical 1.174-order fusion will retain 9,348 correct predictions, while lowering its temperature by one float32 ULP will reduce cross-entropy below 0.18766820755004882 without changing argmax predictions.

INTENDED_EDIT: Average native/reflected probabilities within each crop, apply the verified 1.174-order power mean across crops, and decrease the temperature from 0.753170073032379150390625 to its preceding float32 value.

EVIDENCE: Hierarchical orientation-then-crop fusion achieved the best available score with 9,348 correct and 0.18766820755004882 cross-entropy; separately, the lower-temperature Reference Design 2 outperformed interpolation toward the adjacent higher temperature at identical correctness.

<<<<<<< SEARCH
        fusion_power = 1.125
        for crop, weight in zip(crops, crop_weights):
            native_probabilities = F.softmax(
                self._forward_once(crop), dim=1
            )
            reflected_probabilities = F.softmax(
                self._forward_once(crop.flip(-1)), dim=1
            )
            powered_probabilities = powered_probabilities + weight * (
                native_probabilities.pow(fusion_power)
                + reflected_probabilities.pow(fusion_power)
            )
        probabilities = (
            powered_probabilities / (2.0 * sum(crop_weights))
        ).pow(1.0 / fusion_power)
        probabilities = probabilities / probabilities.sum(
            dim=1,
            keepdim=True,
        )
        return probabilities.clamp_min(
            torch.finfo(probabilities.dtype).tiny
        ).log() / 0.753170073032379150390625
=======
        fusion_power = 1.174
        for crop, weight in zip(crops, crop_weights):
            native_probabilities = F.softmax(
                self._forward_once(crop), dim=1
            )
            reflected_probabilities = F.softmax(
                self._forward_once(crop.flip(-1)), dim=1
            )
            orientation_consensus = 0.5 * (
                native_probabilities + reflected_probabilities
            )
            powered_probabilities = powered_probabilities + (
                2.0 * weight
            ) * orientation_consensus.pow(fusion_power)
        probabilities = (
            powered_probabilities / (2.0 * sum(crop_weights))
        ).pow(1.0 / fusion_power)
        probabilities = probabilities / probabilities.sum(
            dim=1,
            keepdim=True,
        )
        return probabilities.clamp_min(
            torch.finfo(probabilities.dtype).tiny
        ).log() / 0.753170013427734375
>>>>>>> REPLACE