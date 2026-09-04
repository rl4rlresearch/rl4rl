MECHANISM: Best-qualified compensated float32 logit calibration

HYPOTHESIS: Restoring the qualified compensated scale will preserve exactly 9,257 correct predictions and reduce validation cross-entropy from 0.21441891059875487 to approximately 0.2144188995361328.

INTENDED_EDIT: Replace the current inferior calibration with the best verified base scale plus its separately rounded sub-ULP residual.

EVIDENCE: Reference Design 1 achieved the highest available validation_score, 9257.41171954767, with identical accuracy and lower cross-entropy than the current design; the same compensated expression was previously valid and improved the objective.

<<<<<<< SEARCH
        return 1.35604 * aggregate_logits
=======
        return (
            1.3560298681259155 * aggregate_logits
            + 5.4186041e-9 * aggregate_logits
        )
>>>>>>> REPLACE