MECHANISM: Numerically stable affine TTA interpolation

HYPOTHESIS: Re-expressing the unchanged evaluation ensemble with `torch.lerp` will preserve 9,359 correct predictions while reducing cross-entropy below 0.18465710182189943 through different float32 rounding.

INTENDED_EDIT: Compute the same 53.8072967529296875% original / 46.1927032470703125% flipped ensemble as one affine interpolation; retain calibration and training unchanged.

EVIDENCE: Nearby weight and scale probes preserved accuracy but worsened cross-entropy, while slightly higher weights lost one correct prediction; this motivates testing an algebraically equivalent numerical formulation without intentionally moving either calibrated value.

<<<<<<< SEARCH
        return 1.226016 * (
            0.538072967529296875 * logits + 0.461927032470703125 * flipped_logits
        )
=======
        return 1.226016 * torch.lerp(
            flipped_logits,
            logits,
            0.538072967529296875,
        )
>>>>>>> REPLACE