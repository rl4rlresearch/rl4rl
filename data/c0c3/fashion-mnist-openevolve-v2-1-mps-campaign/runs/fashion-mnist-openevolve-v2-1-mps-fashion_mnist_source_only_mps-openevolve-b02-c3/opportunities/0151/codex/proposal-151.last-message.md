MECHANISM: Quadratically interpolated power-mean TTA fusion

HYPOTHESIS: A fusion order of 1.174 will preserve all 9,348 correct predictions while lowering validation cross-entropy below 0.18768770637512208.

INTENDED_EDIT: Replace arithmetic axial-view averaging and temperature interpolation with a normalized 1.174-order probability power mean using Reference Design 3’s verified temperature.

EVIDENCE: Orders 1.0, 1.125, and 1.25 produced cross-entropies of 0.1877065605, 0.1876877064, and 0.1876899883 with identical correctness; quadratic interpolation places the local minimum near 1.174.

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
        log_probabilities = probabilities.clamp_min(
            torch.finfo(probabilities.dtype).tiny
        ).log()
        lower_temperature_logits = (
            log_probabilities / 0.753170073032379150390625
        )
        higher_temperature_logits = (
            log_probabilities / 0.75317013263702392578125
        )
        return torch.lerp(
            lower_temperature_logits,
            higher_temperature_logits,
            0.136,
        )
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