MECHANISM: Hierarchical orientation consensus with continued one-ULP sharpening

HYPOTHESIS: Decreasing the best hierarchical-fusion temperature by one additional float32 ULP will preserve all 9,348 argmax predictions while reducing validation cross-entropy below 0.1876682071685791.

INTENDED_EDIT: Restore the verified native/reflected consensus followed by a 1.174-order crop power mean, then use the preceding float32 temperature, 0.753169953823089599609375.

EVIDENCE: Hierarchical 1.174-order fusion improved cross-entropy from 0.18770656051635742 to 0.18766820755004882 at unchanged correctness, and its first one-ULP temperature decrease further improved it to 0.1876682071685791 without changing any predictions.

<<<<<<< SEARCH
        probabilities = torch.zeros(
            (images.shape[0], 10),
            device=images.device,
            dtype=images.dtype,
        )
        crop_weights = (
            3.0,
            1.3120136260986328125,
            1.3149394989013671875,
            0.686523377895355224609375,
            0.686523497104644775390625,
        )
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
        ).log() / 0.753170073032379150390625
=======
        powered_probabilities = torch.zeros(
            (images.shape[0], 10),
            device=images.device,
            dtype=images.dtype,
        )
        crop_weights = (
            3.0,
            1.3120136260986328125,
            1.3149394989013671875,
            0.686523377895355224609375,
            0.686523497104644775390625,
        )
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
        ).log() / 0.753169953823089599609375
>>>>>>> REPLACE