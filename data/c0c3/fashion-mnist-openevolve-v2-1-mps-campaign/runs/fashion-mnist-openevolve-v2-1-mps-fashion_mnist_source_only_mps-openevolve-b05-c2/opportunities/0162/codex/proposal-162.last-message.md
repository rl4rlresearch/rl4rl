MECHANISM: Directional tetradecic confidence-calibration continuation

HYPOTHESIS: Doubling the centered tetradecic coefficient to −0.000000000000214123855451072 will preserve all 9,192 argmax predictions and reduce validation cross-entropy below 0.22194406433105468.

INTENDED_EDIT: Restore the optimized tridecic term, derive the centered fourteenth-power margin feature, and continue one equal-sized tetradecic step beyond the successful initial probe.

EVIDENCE: Adding a tetradecic coefficient of −0.000000000000107061927725536 reduced cross-entropy from 0.2219440673828125 to 0.22194406433105468 at identical accuracy; prior higher-order searches use an equal-step continuation after a successful initial probe.

<<<<<<< SEARCH
        centered_duodecic = centered_sextic.square()
        confidence_scale = (
=======
        centered_duodecic = centered_sextic.square()
        centered_tridecic = centered_margin * centered_duodecic
        centered_tetradecic = centered_septic.square()
        confidence_scale = (
>>>>>>> REPLACE

<<<<<<< SEARCH
            - 0.0000000000421373651
            * (centered_duodecic - centered_duodecic.mean())
        )
=======
            - 0.0000000000421373651
            * (centered_duodecic - centered_duodecic.mean())
            - 0.000000000000804976900192
            * (centered_tridecic - centered_tridecic.mean())
            - 0.000000000000214123855451072
            * (centered_tetradecic - centered_tetradecic.mean())
        )
>>>>>>> REPLACE