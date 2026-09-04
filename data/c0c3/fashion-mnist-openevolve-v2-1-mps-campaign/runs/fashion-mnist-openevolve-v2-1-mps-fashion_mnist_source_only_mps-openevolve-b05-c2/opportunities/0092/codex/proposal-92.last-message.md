MECHANISM: Quadratic-vertex confidence calibration

HYPOTHESIS: A centered quadratic coefficient of −0.003125 will preserve all 9,192 correct predictions while reducing validation cross-entropy below 0.22200205307006837.

INTENDED_EDIT: Replace the current linear margin calibration with the best verified linear slope plus a locally optimized centered quadratic correction.

EVIDENCE: Coefficients 0, −0.001, −0.002, and −0.003 yielded cross-entropies 0.2220624084, 0.2220299664, 0.2220097668, and 0.2220020531 with identical accuracy; quadratic interpolation places the minimum near −0.003125.

<<<<<<< SEARCH
        top_two = normalized_logits.topk(2, dim=1).values
        margin = top_two[:, :1] - top_two[:, 1:2]
        confidence_scale = (10500.0 / 9564.0) + 0.01 * (
            margin - margin.mean()
        )
        confidence_scale = confidence_scale.clamp(0.75, 1.5)
=======
        top_two = normalized_logits.topk(2, dim=1).values
        margin = top_two[:, :1] - top_two[:, 1:2]
        centered_margin = margin - margin.mean()
        centered_square = centered_margin.square()
        confidence_scale = (
            (10500.0 / 9564.0)
            + 0.017065 * centered_margin
            - 0.003125 * (centered_square - centered_square.mean())
        )
        confidence_scale = confidence_scale.clamp(0.75, 1.5)
>>>>>>> REPLACE