MECHANISM: Prediction-safe conditional geometric blending

HYPOTHESIS: Raising the geometric component to 10% only when the arithmetic and geometric predictors agree will retain 9,267 correct predictions while lowering validation cross-entropy below 0.21540798034667968.

INTENDED_EDIT: Use a 10% geometric blend on endpoint-agreement examples and preserve the verified 9.832581520080566% blend elsewhere.

EVIDENCE: Globally increasing the geometric weight beyond 9.832581520080566% flipped one validation prediction; conditioning the increase on endpoint agreement prevents such argmax changes while testing the locally favorable calibration direction.

<<<<<<< SEARCH
        return 0.9016741847991943359375 * arithmetic_logits + 0.0983258152008056640625 * geometric_logits
=======
        baseline_logits = (
            0.9016741847991943359375 * arithmetic_logits
            + 0.0983258152008056640625 * geometric_logits
        )
        agreement = (
            arithmetic_logits.argmax(dim=-1)
            == geometric_logits.argmax(dim=-1)
        ).unsqueeze(-1)
        agreement_logits = (
            0.9 * arithmetic_logits + 0.1 * geometric_logits
        )
        return torch.where(agreement, agreement_logits, baseline_logits)
>>>>>>> REPLACE