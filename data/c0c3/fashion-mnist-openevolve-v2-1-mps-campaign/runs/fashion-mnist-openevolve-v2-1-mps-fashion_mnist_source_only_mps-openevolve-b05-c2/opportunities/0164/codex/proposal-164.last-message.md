MECHANISM: Directional tetradecic confidence-calibration continuation

HYPOTHESIS: Advancing the centered tetradecic coefficient to −0.000000000000428247710902144 will preserve all 9,192 argmax predictions and reduce validation cross-entropy below 0.2219440544128418.

INTENDED_EDIT: Restore the optimized tridecic coefficient, add the centered fourteenth-power margin feature, and continue the tetradecic coefficient by one equal-sized negative step.

EVIDENCE: Moving the tetradecic coefficient from −0.000000000000214123855451072 to −0.000000000000321185783176608 reduced cross-entropy from 0.22194405670166015 to 0.2219440544128418 without changing accuracy, motivating one further continuation probe.

<<<<<<< SEARCH
        centered_duodecic = centered_sextic.square()
        centered_tridecic = centered_margin * centered_duodecic
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
            + 0.0000000000056042695583
            * (centered_tridecic - centered_tridecic.mean())
=======
            - 0.0000000000421373651
            * (centered_duodecic - centered_duodecic.mean())
            - 0.000000000000804976900192
            * (centered_tridecic - centered_tridecic.mean())
            - 0.000000000000428247710902144
            * (centered_tetradecic - centered_tetradecic.mean())
>>>>>>> REPLACE