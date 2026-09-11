MECHANISM: Float32 adjacent-temperature logit interpolation

HYPOTHESIS: Interpolating 13.6% from Reference Design 2’s temperature-scaled logits toward the next higher float32 temperature will preserve 9,348 correct predictions while reducing validation cross-entropy below 0.18770656051635742.

INTENDED_EDIT: Restore Reference Design 2 as the lower interpolation endpoint and return a float32 linear interpolation between logits produced by the two adjacent temperatures.

EVIDENCE: Reference Design 2 has the lowest verified cross-entropy, while both adjacent float32 temperatures were worse; quadratic interpolation estimated the optimum 0.136 ULP higher. The earlier float64 interpolation could not be verified, and reciprocal multiplication did not reproduce the desired arithmetic.

<<<<<<< SEARCH
        return probabilities.clamp_min(
            torch.finfo(probabilities.dtype).tiny
        ).log() / 0.75317013263702392578125
=======
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