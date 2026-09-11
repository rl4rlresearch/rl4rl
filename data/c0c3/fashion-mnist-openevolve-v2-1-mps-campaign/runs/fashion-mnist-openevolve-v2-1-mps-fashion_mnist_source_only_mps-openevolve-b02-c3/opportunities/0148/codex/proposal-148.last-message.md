MECHANISM: Mild higher-order power-mean TTA fusion

HYPOTHESIS: A 1.125-order power mean will preserve all 9,348 correct predictions while lowering validation cross-entropy below 0.18770656051635742 by modestly favoring crop-specific high-confidence evidence.

INTENDED_EDIT: Restore equal native/reflected weighting and the best verified temperature, then replace arithmetic probability averaging with a normalized 1.125-order weighted power mean over the proven ten axial views.

EVIDENCE: Blending 12.5% toward geometric consensus preserved correctness but worsened cross-entropy to 0.18773939895629882; this supplies directional evidence that moving the fusion rule oppositely, above the arithmetic mean, is the most informative local test.

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
        native_weight = 0.99999988079071044921875
        reflected_weight = 1.00000011920928955078125
        for crop, weight in zip(crops, crop_weights):
            probabilities = probabilities + (
                weight * native_weight
            ) * F.softmax(
                self._forward_once(crop), dim=1
            )
            probabilities = probabilities + (
                weight * reflected_weight
            ) * F.softmax(
                self._forward_once(crop.flip(-1)), dim=1
            )
        probabilities = probabilities / (
            (native_weight + reflected_weight) * sum(crop_weights)
        )
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
>>>>>>> REPLACE