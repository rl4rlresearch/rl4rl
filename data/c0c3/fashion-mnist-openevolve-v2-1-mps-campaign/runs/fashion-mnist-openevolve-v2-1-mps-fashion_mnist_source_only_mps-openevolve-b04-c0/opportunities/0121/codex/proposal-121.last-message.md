MECHANISM: Split-constant float32 temperature calibration

HYPOTHESIS: Compensated float32 scaling will preserve 9,359 predictions while reducing cross-entropy below 0.18465710182189943 by approximating the intended 1.226016 scale more accurately without unsupported float64 logits.

INTENDED_EDIT: Materialize the unchanged ensemble, then apply 1.226016 as high and low float32 components while retaining the model’s output dtype.

EVIDENCE: The current ensemble achieved 9,359 correct at 0.18465710182189943; the subsequent float64 calibration attempt could not be verified, motivating an equivalent precision refinement that remains float32-compatible.

<<<<<<< SEARCH
        return 1.226016 * (
            0.538072967529296875 * logits + 0.461927032470703125 * flipped_logits
        )
=======
        ensemble_logits = (
            0.538072967529296875 * logits
            + 0.461927032470703125 * flipped_logits
        )
        return (
            1.22601592540740966796875 * ensemble_logits
            + 0.00000007459259033203125 * ensemble_logits
        )
>>>>>>> REPLACE