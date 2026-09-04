MECHANISM: Quadratic-informed order-locked temperature refinement

HYPOTHESIS: A 75.9%-of-normalization effective denominator will retain 9,192 correct predictions and reduce validation cross-entropy below 0.2223825225830078.

INTENDED_EDIT: Anchor evaluation logits to the proven 84% denominator, max-center them to preserve class ordering, then sharpen to an effective 75.9% denominator.

EVIDENCE: Cross-entropy improved from 0.22403448944091797 at 84% to 0.2223825225830078 at 76%, but worsened to 0.224783793258667 at 68%; these measurements place the local calibration minimum near 75.9%.

<<<<<<< SEARCH
        return logit_sum / 15.0999375
=======
        normalized_logits = logit_sum / 13.2124453125
        normalized_logits = normalized_logits - normalized_logits.amax(
            dim=1, keepdim=True
        )
        return normalized_logits * (280.0 / 253.0)
>>>>>>> REPLACE