MECHANISM: Centered-heptadecic confidence calibration

HYPOTHESIS: A positive heptadecic coefficient of +0.00000000000000291373587358294 on the best verified pentadecic design will preserve all 9,192 argmax predictions and reduce validation cross-entropy below 0.22194401245117187.

INTENDED_EDIT: Restore the best pentadecic-only calibration, remove the unsuccessful hexadecic term, and add a bounded positive centered seventeenth-power margin feature.

EVIDENCE: The pentadecic-only design achieved the best observed cross-entropy, while both tested hexadecic directions were worse; the positive pentadecic direction was beneficial, motivating a smaller same-sign probe in the next odd-order basis.

<<<<<<< SEARCH
        centered_pentadecic = centered_margin * centered_tetradecic
        centered_hexadecic = centered_octic.square()
        confidence_scale = (
=======
        centered_pentadecic = centered_margin * centered_tetradecic
        centered_hexadecic = centered_octic.square()
        centered_heptadecic = centered_margin * centered_hexadecic
        confidence_scale = (
>>>>>>> REPLACE

<<<<<<< SEARCH
            + 0.00000000000016472021446
            * (centered_pentadecic - centered_pentadecic.mean())
            - 0.00000000000002190778852318
            * (centered_hexadecic - centered_hexadecic.mean())
=======
            + 0.00000000000016472021446
            * (centered_pentadecic - centered_pentadecic.mean())
            + 0.00000000000000291373587358294
            * (centered_heptadecic - centered_heptadecic.mean())
>>>>>>> REPLACE