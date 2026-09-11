MECHANISM: Arithmetic–geometric TTA consensus blending

HYPOTHESIS: Blending 12.5% geometric consensus into the successful axial probability ensemble will exceed 9,348 correct predictions by suppressing isolated high-confidence crop errors while preserving the proven view set.

INTENDED_EDIT: Retain the five axial crops and equal reflection weighting, blend their arithmetic and geometric probability ensembles, and restore the best verified 0.136 adjacent-temperature interpolation.

EVIDENCE: Axial translation TTA improved correctness from 9,344 to 9,348, while adding diagonal views reduced it to 9,335; changing how the proven axial views are fused is therefore a more informative accuracy probe than adding views or continuing temperature-only refinements.

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
            0.25,
        )
=======
        probability_sum = torch.zeros(
            (images.shape[0], 10),
            device=images.device,
            dtype=images.dtype,
        )
        log_probability_sum = torch.zeros_like(probability_sum)
        crop_weights = (
            3.0,
            1.3120136260986328125,
            1.3149394989013671875,
            0.686523377895355224609375,
            0.686523497104644775390625,
        )
        for crop, weight in zip(crops, crop_weights):
            native_logits = self._forward_once(crop)
            reflected_logits = self._forward_once(crop.flip(-1))
            probability_sum = probability_sum + weight * (
                F.softmax(native_logits, dim=1)
                + F.softmax(reflected_logits, dim=1)
            )
            log_probability_sum = log_probability_sum + weight * (
                F.log_softmax(native_logits, dim=1)
                + F.log_softmax(reflected_logits, dim=1)
            )

        normalizer = 2.0 * sum(crop_weights)
        arithmetic_probabilities = probability_sum / normalizer
        geometric_probabilities = F.softmax(
            log_probability_sum / normalizer,
            dim=1,
        )
        probabilities = torch.lerp(
            arithmetic_probabilities,
            geometric_probabilities,
            0.125,
        )
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
>>>>>>> REPLACE