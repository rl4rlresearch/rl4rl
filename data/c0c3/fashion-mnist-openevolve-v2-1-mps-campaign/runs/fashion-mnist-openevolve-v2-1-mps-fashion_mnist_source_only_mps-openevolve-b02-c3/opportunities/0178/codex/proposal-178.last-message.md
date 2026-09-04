MECHANISM: Reciprocal-quantized temperature calibration

HYPOTHESIS: Replacing interpolated temperature division with the verified float32 reciprocal multiplier will preserve 9,348 correct predictions and lower cross-entropy from 0.18766820449829102 to approximately 0.18766820373535156.

INTENDED_EDIT: Keep the verified hierarchical 1.174-order fusion unchanged and apply calibration through the exact reciprocal-multiplication path from Reference Design 2.

EVIDENCE: Reference Design 2 used the same architecture, training, and fusion, achieving the best available validation_score with 9,348 correct and cross-entropy 0.18766820373535156.

<<<<<<< SEARCH
        log_probabilities = probabilities.clamp_min(
            torch.finfo(probabilities.dtype).tiny
        ).log()
        best_temperature_logits = (
            log_probabilities / 0.753169953823089599609375
        )
        lower_temperature_logits = (
            log_probabilities / 0.75316989421844482421875
        )
        return torch.lerp(
            best_temperature_logits,
            lower_temperature_logits,
            3.0 / 22.0,
        )
=======
        return probabilities.clamp_min(
            torch.finfo(probabilities.dtype).tiny
        ).log() * 1.32772159576416015625
>>>>>>> REPLACE