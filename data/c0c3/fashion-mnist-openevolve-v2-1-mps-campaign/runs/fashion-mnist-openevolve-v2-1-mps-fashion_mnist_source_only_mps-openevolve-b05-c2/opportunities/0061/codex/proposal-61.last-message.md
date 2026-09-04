MECHANISM: Double-precision order-preserving logit calibration

HYPOTHESIS: Applying the proven 76.512% calibration in float64 will preserve all 9,192 predictions while reducing validation cross-entropy below 0.22237433319091796.

INTENDED_EDIT: Cast accumulated ensemble logits to float64 before normalization, max-centering, and temperature scaling.

EVIDENCE: The 76.512% denominator produced the best verified cross-entropy, while finer temperature probes were worse; this makes calibration arithmetic precision the most informative remaining variable.

<<<<<<< SEARCH
        normalized_logits = logit_sum / 13.2124453125
        normalized_logits = normalized_logits - normalized_logits.amax(
            dim=1, keepdim=True
        )
        return normalized_logits * (10500.0 / 9564.0)
=======
        normalized_logits = logit_sum.to(torch.float64) / 13.2124453125
        normalized_logits = normalized_logits - normalized_logits.amax(
            dim=1, keepdim=True
        )
        return normalized_logits * (10500.0 / 9564.0)
>>>>>>> REPLACE