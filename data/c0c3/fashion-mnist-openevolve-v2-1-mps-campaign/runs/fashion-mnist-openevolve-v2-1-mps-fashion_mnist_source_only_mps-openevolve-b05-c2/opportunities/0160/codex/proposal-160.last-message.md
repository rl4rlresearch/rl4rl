MECHANISM: Quadratic-vertex tridecic confidence calibration

HYPOTHESIS: A centered tridecic coefficient of −0.000000000000804976900192 will preserve all 9,192 argmax predictions and reduce validation cross-entropy below 0.22194406814575196.

INTENDED_EDIT: Restore the optimized undecic and duodecic coefficients, add the centered thirteenth-power margin feature, and set its coefficient to the quadratic optimum fitted from the two symmetric tridecic probes and the zero-coefficient baseline.

EVIDENCE: Tridecic coefficients −0.0000000000056042695583, 0, and +0.0000000000056042695583 produced cross-entropies 0.22194410552978516, 0.22194406814575196, and 0.22194413566589355 at identical accuracy; quadratic interpolation places the minimum slightly negative at approximately −0.000000000000804976900192.

<<<<<<< SEARCH
        centered_undecic = centered_margin * centered_decic
        confidence_scale = (
=======
        centered_undecic = centered_margin * centered_decic
        centered_duodecic = centered_sextic.square()
        centered_tridecic = centered_margin * centered_duodecic
        confidence_scale = (
>>>>>>> REPLACE

<<<<<<< SEARCH
            - 0.000000001716523096 * (centered_decic - centered_decic.mean())
            - 0.0000000002283 * (centered_undecic - centered_undecic.mean())
        )
=======
            - 0.000000001716523096 * (centered_decic - centered_decic.mean())
            - 0.0000000001178008529
            * (centered_undecic - centered_undecic.mean())
            - 0.0000000000421373651
            * (centered_duodecic - centered_duodecic.mean())
            - 0.000000000000804976900192
            * (centered_tridecic - centered_tridecic.mean())
        )
>>>>>>> REPLACE