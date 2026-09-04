MECHANISM: Reciprocal-quantized temperature calibration

HYPOTHESIS: Using the verified float32 reciprocal multiplier will preserve all 9,348 correct predictions while reducing validation cross-entropy from 0.18766820449829102 to approximately 0.18766820373535156.

INTENDED_EDIT: Replace sub-ULP interpolation between temperature divisions with the single reciprocal multiplication from the highest-scoring verified design.

EVIDENCE: Reference Design 2 achieved the best available validation score, retaining 9,348 correct predictions and improving cross-entropy to 0.18766820373535156; the current interpolation has identical correctness but slightly higher cross-entropy.

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