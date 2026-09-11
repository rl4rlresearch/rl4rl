MECHANISM: Quadratic-fit order-preserving temperature calibration

HYPOTHESIS: An effective denominator of 76.74% will retain 9,192 correct predictions while reducing validation cross-entropy below 0.2223825225830078.

INTENDED_EDIT: Anchor logits to the proven 84% denominator, max-center them, then sharpen by 84/76.74 without changing class ordering.

EVIDENCE: Cross-entropy was 0.224783793258667 at 68%, 0.2223825225830078 at 76%, and 0.22403448944091797 at 84%; a quadratic fit to these evenly spaced measurements places the minimum near 76.74%, while the 75.9% refinement was slightly worse.

<<<<<<< SEARCH
        return logit_sum / 15.7291015625
=======
        normalized_logits = logit_sum / 13.2124453125
        normalized_logits = normalized_logits - normalized_logits.amax(
            dim=1, keepdim=True
        )
        return normalized_logits * (4200.0 / 3837.0)
>>>>>>> REPLACE