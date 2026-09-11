MECHANISM: Adjacent-float logit interpolation

HYPOTHESIS: Equal interpolation between the two adjacent multiplication coefficients tied at the best verified cross-entropy will preserve all 9,192 predictions while producing validation cross-entropy below 0.22237433319091796.

INTENDED_EDIT: Replace reciprocal division with an elementwise midpoint blend of logits produced by Reference Designs 3 and 1.

EVIDENCE: The rational 10500/9564 coefficient and its immediately larger float32 neighbor both achieved the best observed cross-entropy, while coefficients outside that pair were worse, indicating that an effective midpoint is the most informative remaining calibration probe.

<<<<<<< SEARCH
        return normalized_logits / (1.0 / 1.0978671312332153)
=======
        lower_scaled = normalized_logits * (10500.0 / 9564.0)
        upper_scaled = normalized_logits * 1.0978671312332153
        return torch.lerp(lower_scaled, upper_scaled, 0.5)
>>>>>>> REPLACE